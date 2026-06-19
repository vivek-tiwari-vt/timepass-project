"""ConceptBridge FastAPI backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import CONFIG
from explain import explain_path, format_detail_view
from search.semantic_bfs import semantic_bidirectional_search
from search.bfs import bidirectional_bfs
from sources.wikipedia import WikipediaSource

app = FastAPI(title="ConceptBridge", version="1.0")

# Serve static frontend files
static_dir = Path(__file__).parent.parent / "frontend" / "dist"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class BridgeRequest(BaseModel):
    start: str
    end: str
    semantic: bool = True
    explain: bool = True
    provider: Optional[str] = None   # "anthropic" | "deepseek" | None (uses config default)


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
    graph: dict          # cytoscape-compatible JSON


def _build_graph(path: list[str]) -> dict:
    """Build a cytoscape.js compatible element list from the path."""
    nodes = []
    edges = []
    seen_nodes: set[str] = set()

    for i, title in enumerate(path):
        if title not in seen_nodes:
            nodes.append({
                "data": {
                    "id": title,
                    "label": title,
                    "on_path": True,
                    "index": i,
                }
            })
            seen_nodes.add(title)
        if i < len(path) - 1:
            edges.append({
                "data": {
                    "id": f"{title}->{path[i+1]}",
                    "source": title,
                    "target": path[i + 1],
                    "on_path": True,
                }
            })

    return {"nodes": nodes, "edges": edges}


@app.post("/api/bridge", response_model=BridgeResponse)
async def bridge(req: BridgeRequest):
    source = WikipediaSource()
    try:
        start_concept, end_concept = await __import__("asyncio").gather(
            source.resolve_concept(req.start),
            source.resolve_concept(req.end),
        )

        search_fn = semantic_bidirectional_search if req.semantic else bidirectional_bfs
        result = await search_fn(
            source,
            start_concept.title,
            end_concept.title,
            trace_file=CONFIG.trace_file,
        )

        if result is None:
            raise HTTPException(status_code=404, detail="No path found within budget")

        hops_data: list[HopData] = []
        if req.explain:
            if req.provider:
                CONFIG.llm_provider = req.provider
            explanations = await explain_path(source, result.path)
            hops_data = [HopData(from_title=e.from_title, to_title=e.to_title, sentence=e.sentence) for e in explanations]

        graph = _build_graph(result.path)

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
            graph=graph,
        )
    finally:
        await source.close()


@app.get("/api/export/json")
async def export_json(start: str, end: str, semantic: bool = True):
    """Re-run and return raw JSON suitable for further processing."""
    req = BridgeRequest(start=start, end=end, semantic=semantic, explain=False)
    result = await bridge(req)
    return result


@app.get("/")
async def index():
    index_file = Path(__file__).parent.parent / "frontend" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "ConceptBridge API running. POST to /api/bridge"})
