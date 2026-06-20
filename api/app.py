"""ConceptBridge FastAPI backend."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import CONFIG
from explain import explain_path, format_detail_view
from search.semantic_bfs import semantic_bidirectional_search
from search.bfs import bidirectional_bfs
from sources.wikipedia import WikipediaSource

app = FastAPI(title="ConceptBridge", version="1.0")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class BridgeRequest(BaseModel):
    start: str
    end: str
    semantic: bool = True
    explain: bool = True
    provider: Optional[str] = None   # "anthropic" | "deepseek" | None (config default)


class HopData(BaseModel):
    from_title: str
    to_title: str
    sentence: str


class BridgeResponse(BaseModel):
    start_resolved: str
    end_resolved: str
    start_alternatives: list[str]
    end_alternatives: list[str]
    path: list[str]
    hops: list[HopData]
    nodes_explored: int
    elapsed: float
    met_at: str
    graph: dict


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_graph(path: list[str]) -> dict:
    nodes, edges, seen = [], [], set()
    for i, title in enumerate(path):
        if title not in seen:
            nodes.append({"data": {"id": title, "label": title, "on_path": True, "index": i}})
            seen.add(title)
        if i < len(path) - 1:
            edges.append({"data": {"id": f"{title}->{path[i+1]}", "source": title, "target": path[i + 1], "on_path": True}})
    return {"nodes": nodes, "edges": edges}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------
# SSE streaming endpoint  /api/bridge/stream
# ------------------------------------------------------------------

@app.get("/api/bridge/stream")
async def bridge_stream(
    start: str,
    end: str,
    semantic: bool = True,
    explain: bool = True,
    provider: Optional[str] = None,
):
    """Server-Sent Events endpoint — streams live search progress to the browser."""
    queue: asyncio.Queue = asyncio.Queue()

    async def pipeline():
        source = WikipediaSource()
        try:
            await queue.put({"type": "status", "message": "Resolving concepts…"})
            try:
                start_concept, end_concept = await asyncio.gather(
                    source.resolve_concept(start),
                    source.resolve_concept(end),
                )
            except ValueError as e:
                await queue.put({"type": "error", "message": str(e)})
                return

            await queue.put({
                "type": "resolved",
                "start": start_concept.title,
                "end": end_concept.title,
                "start_url": start_concept.url,
                "end_url": end_concept.url,
                "start_alts": start_concept.alternatives[:3],
                "end_alts": end_concept.alternatives[:3],
            })

            search_fn = semantic_bidirectional_search if semantic else bidirectional_bfs
            await queue.put({"type": "status", "message": f"Running {'semantic ' if semantic else ''}bidirectional BFS…"})

            result = await search_fn(
                source,
                start_concept.title,
                end_concept.title,
                trace_file=CONFIG.trace_file,
                event_queue=queue,
            )

            if result is None:
                await queue.put({"type": "error", "message": "No path found within node budget. Try increasing max_depth."})
                return

            await queue.put({
                "type": "path_found",
                "path": result.path,
                "met_at": result.met_at,
                "nodes_explored": result.nodes_explored,
                "elapsed": round(result.elapsed, 2),
            })

            if explain:
                if provider:
                    CONFIG.llm_provider = provider
                await queue.put({"type": "status", "message": f"Generating hop explanations via {CONFIG.llm_provider}…"})
                explanations = await explain_path(source, result.path)
                for i, exp in enumerate(explanations):
                    await queue.put({
                        "type": "hop",
                        "index": i,
                        "from_title": exp.from_title,
                        "to_title": exp.to_title,
                        "sentence": exp.sentence,
                    })

            await queue.put({
                "type": "done",
                "nodes_explored": result.nodes_explored,
                "elapsed": round(result.elapsed, 2),
                "path_length": len(result.path) - 1,
            })

        except Exception as e:
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await source.close()
            await queue.put(None)   # sentinel

    task = asyncio.create_task(pipeline())

    async def event_generator() -> AsyncGenerator[str, None]:
        # keepalive comment so proxy/browser doesn't close idle connection
        yield ": keepalive\n\n"
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=90)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                yield _sse(event)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ------------------------------------------------------------------
# Standard POST endpoint  /api/bridge  (non-streaming, for API clients)
# ------------------------------------------------------------------

@app.post("/api/bridge", response_model=BridgeResponse)
async def bridge(req: BridgeRequest):
    source = WikipediaSource()
    try:
        start_concept, end_concept = await asyncio.gather(
            source.resolve_concept(req.start),
            source.resolve_concept(req.end),
        )
        search_fn = semantic_bidirectional_search if req.semantic else bidirectional_bfs
        result = await search_fn(source, start_concept.title, end_concept.title, trace_file=CONFIG.trace_file)
        if result is None:
            raise HTTPException(status_code=404, detail="No path found within budget")

        hops_data: list[HopData] = []
        if req.explain:
            if req.provider:
                CONFIG.llm_provider = req.provider
            explanations = await explain_path(source, result.path)
            hops_data = [HopData(from_title=e.from_title, to_title=e.to_title, sentence=e.sentence) for e in explanations]

        return BridgeResponse(
            start_resolved=start_concept.title,
            end_resolved=end_concept.title,
            start_alternatives=start_concept.alternatives,
            end_alternatives=end_concept.alternatives,
            path=result.path,
            hops=hops_data,
            nodes_explored=result.nodes_explored,
            elapsed=result.elapsed,
            met_at=result.met_at,
            graph=_build_graph(result.path),
        )
    finally:
        await source.close()


# ------------------------------------------------------------------
# Static / index
# ------------------------------------------------------------------

@app.get("/")
async def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "ConceptBridge API — POST /api/bridge or GET /api/bridge/stream"})
