"""MediaWiki-backed GraphSource for Wikipedia."""

import asyncio
import re
import httpx

from config import CONFIG
from sources.base import GraphSource, ResolvedConcept


async def _get_with_retry(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    for attempt in range(5):
        resp = await client.get(url, params=params)
        if resp.status_code == 429:
            wait = 2 ** attempt
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


class WikipediaSource(GraphSource):
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    def _client_instance(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": CONFIG.wikipedia_user_agent},
                timeout=30,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # GraphSource interface
    # ------------------------------------------------------------------

    async def resolve_concept(self, concept: str) -> ResolvedConcept:
        client = self._client_instance()
        params = {
            "action": "opensearch",
            "search": concept,
            "limit": 5,
            "namespace": 0,
            "format": "json",
        }
        resp = await _get_with_retry(client, CONFIG.wikipedia_api_url, params)
        data = resp.json()
        # opensearch returns [query, [titles], [descriptions], [urls]]
        titles = data[1]
        urls = data[3]
        if not titles:
            raise ValueError(f"No Wikipedia article found for concept: {concept!r}")
        return ResolvedConcept(
            title=titles[0],
            url=urls[0],
            alternatives=titles[1:],
        )

    async def get_outgoing_links(self, title: str) -> list[str]:
        """Fetch all article-namespace links from `title` (batched)."""
        return await self._paginated_prop(title, prop="links", pl_key="links", title_key="title")

    async def get_incoming_links(self, title: str) -> list[str]:
        """Fetch all article-namespace backlinks to `title` (batched)."""
        client = self._client_instance()
        results: list[str] = []
        blcontinue: str | None = None

        while True:
            params: dict = {
                "action": "query",
                "list": "backlinks",
                "bltitle": title,
                "blnamespace": 0,
                "bllimit": CONFIG.wikipedia_batch_size,
                "blfilterredir": "nonredirects",
                "format": "json",
            }
            if blcontinue:
                params["blcontinue"] = blcontinue

            resp = await _get_with_retry(client, CONFIG.wikipedia_api_url, params)
            data = resp.json()

            for item in data.get("query", {}).get("backlinks", []):
                results.append(item["title"])

            blcontinue = data.get("continue", {}).get("blcontinue")
            if not blcontinue:
                break

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _paginated_prop(
        self, title: str, prop: str, pl_key: str, title_key: str
    ) -> list[str]:
        """Generic paginated prop fetch (used for outgoing links)."""
        client = self._client_instance()
        results: list[str] = []
        plcontinue: str | None = None

        while True:
            params: dict = {
                "action": "query",
                "titles": title,
                "prop": prop,
                "pllimit": CONFIG.wikipedia_batch_size,
                "plnamespace": 0,
                "format": "json",
            }
            if plcontinue:
                params["plcontinue"] = plcontinue

            resp = await _get_with_retry(client, CONFIG.wikipedia_api_url, params)
            data = resp.json()

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                for link in page.get(pl_key, []):
                    results.append(link[title_key])

            plcontinue = data.get("continue", {}).get("plcontinue")
            if not plcontinue:
                break
            if CONFIG.max_links_per_page and len(results) >= CONFIG.max_links_per_page:
                break

        return results[:CONFIG.max_links_per_page] if CONFIG.max_links_per_page else results

    async def get_page_intro(self, title: str) -> str:
        """Return first ~500 chars of article extract (for LLM context)."""
        client = self._client_instance()
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "exchars": 500,
            "explaintext": True,
            "format": "json",
        }
        resp = await _get_with_retry(client, CONFIG.wikipedia_api_url, params)
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "")
        return ""


# ------------------------------------------------------------------
# TODO: OpenWebSource
# ------------------------------------------------------------------

class OpenWebSource(GraphSource):
    """Stub — forward-only (no backlinks on the open web).

    Resolution: DuckDuckGo / Google Custom Search to pick canonical URL.
    Outgoing links: fetch URL, parse <a href> with BeautifulSoup, filter same-domain or
        content-looking URLs.
    Incoming links: NOT SUPPORTED on the open web without a paid index (e.g. Ahrefs).
        Bidirectional BFS degrades to forward-only greedy from start.
    Paths will be fuzzier and noisier than Wikipedia paths.
    """

    async def resolve_concept(self, concept: str) -> ResolvedConcept:
        raise NotImplementedError("OpenWebSource not yet implemented")

    async def get_outgoing_links(self, title: str) -> list[str]:
        raise NotImplementedError("OpenWebSource not yet implemented")

    async def get_incoming_links(self, title: str) -> list[str]:
        raise NotImplementedError("Backlinks not available on the open web")
