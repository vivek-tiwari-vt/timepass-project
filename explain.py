"""LLM-powered hop-by-hop explanations.

Supports two providers, selected by CONFIG.llm_provider:
  "anthropic" — Anthropic Messages API  (requires ANTHROPIC_API_KEY)
  "deepseek"  — DeepSeek Chat API, OpenAI-compatible (requires DEEPSEEK_API_KEY)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Protocol

import httpx

from config import CONFIG
from sources.wikipedia import WikipediaSource


@dataclass
class HopExplanation:
    from_title: str
    to_title: str
    sentence: str


# ------------------------------------------------------------------
# Provider protocol
# ------------------------------------------------------------------

class LLMClient(Protocol):
    async def complete(self, prompt: str) -> str: ...


# ------------------------------------------------------------------
# Anthropic provider
# ------------------------------------------------------------------

class AnthropicClient:
    def __init__(self, api_key: str):
        import anthropic as _anthropic
        self._client = _anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt: str) -> str:
        msg = await self._client.messages.create(
            model=CONFIG.llm_model,
            max_tokens=CONFIG.llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()


# ------------------------------------------------------------------
# DeepSeek provider  (OpenAI-compatible REST)
# ------------------------------------------------------------------

class DeepSeekClient:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._http = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )

    async def complete(self, prompt: str) -> str:
        payload = {
            "model": CONFIG.deepseek_model,
            "max_tokens": CONFIG.llm_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = await self._http.post(CONFIG.deepseek_api_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    async def aclose(self):
        await self._http.aclose()


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def _make_client() -> tuple[LLMClient | None, str]:
    """Return (client, error_message). error_message is empty on success."""
    provider = CONFIG.llm_provider.lower()

    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return None, "[LLM explanation unavailable — set ANTHROPIC_API_KEY]"
        return AnthropicClient(key), ""

    if provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            return None, "[LLM explanation unavailable — set DEEPSEEK_API_KEY]"
        return DeepSeekClient(key), ""

    return None, f"[Unknown llm_provider '{provider}' — choose 'anthropic' or 'deepseek']"


# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------

def _build_prompt(from_title: str, from_intro: str, to_title: str, to_intro: str) -> str:
    return (
        f'You are explaining a conceptual link in a Wikipedia path.\n\n'
        f'Source article: "{from_title}"\n'
        f'Context: {from_intro[:300]}\n\n'
        f'Target article: "{to_title}"\n'
        f'Context: {to_intro[:300]}\n\n'
        f'Write exactly ONE concise sentence (max 30 words) that explains the conceptual '
        f'bridge from "{from_title}" to "{to_title}". No preamble, just the sentence.'
    )


async def _explain_hop(
    llm: LLMClient,
    source: WikipediaSource,
    from_title: str,
    to_title: str,
) -> HopExplanation:
    from_intro, to_intro = await asyncio.gather(
        source.get_page_intro(from_title),
        source.get_page_intro(to_title),
    )
    prompt = _build_prompt(from_title, from_intro, to_title, to_intro)
    sentence = await llm.complete(prompt)
    return HopExplanation(from_title=from_title, to_title=to_title, sentence=sentence)


async def explain_path(
    source: WikipediaSource,
    path: list[str],
) -> list[HopExplanation]:
    """Generate a one-sentence bridge explanation for every hop in `path`."""
    client, err = _make_client()

    if client is None:
        return [
            HopExplanation(from_title=path[i], to_title=path[i + 1], sentence=err)
            for i in range(len(path) - 1)
        ]

    tasks = [_explain_hop(client, source, path[i], path[i + 1]) for i in range(len(path) - 1)]
    results = await asyncio.gather(*tasks)

    if isinstance(client, DeepSeekClient):
        await client.aclose()

    return list(results)


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------

def format_detail_view(path: list[str], explanations: list[HopExplanation], stats: dict) -> str:
    provider = CONFIG.llm_provider
    lines = ["", "═" * 60, f"  ConceptBridge — Path Detail  [{provider}]", "═" * 60, ""]
    for i, title in enumerate(path):
        lines.append(f"  {i}. {title}")
        if i < len(explanations):
            lines.append(f"       └─ {explanations[i].sentence}")
            lines.append("")
    lines.append("─" * 60)
    lines.append(f"  Path length   : {len(path) - 1} hop(s)")
    lines.append(f"  Nodes explored: {stats.get('nodes_explored', '?')}")
    lines.append(f"  Wall-clock    : {stats.get('elapsed', 0):.2f}s")
    lines.append(f"  Met at node   : {stats.get('met_at', '?')}")
    lines.append(f"  LLM provider  : {provider}")
    lines.append("═" * 60)
    return "\n".join(lines)
