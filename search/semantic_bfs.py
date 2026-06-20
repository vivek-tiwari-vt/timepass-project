"""Bidirectional best-first search with semantic frontier ranking.

On top of plain BFS, each candidate node is ranked by cosine similarity
of its title embedding toward the opposite target.  The top `beam_width`
candidates are expanded first, making the search more directed.

Falls back gracefully if sentence-transformers is not installed.
Pass an asyncio.Queue as `event_queue` to receive live progress events.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from config import CONFIG
from sources.base import GraphSource
from search.bfs import _is_hub, _expand_frontier, SearchResult, _reconstruct_path


# ------------------------------------------------------------------
# Embedding cache
# ------------------------------------------------------------------

_embedding_cache: dict[str, list[float]] = {}
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(CONFIG.embedding_model)
        print(f"[semantic] loaded embedding model: {CONFIG.embedding_model}")
    except ImportError:
        _model = False
        print("[semantic] sentence-transformers not installed; falling back to BFS order")
    return _model


def _embed(titles: list[str]) -> dict[str, list[float]]:
    model = _load_model()
    if not model:
        return {}
    to_encode = [t for t in titles if t not in _embedding_cache]
    if to_encode:
        vecs = model.encode(to_encode, normalize_embeddings=True).tolist()
        for title, vec in zip(to_encode, vecs):
            _embedding_cache[title] = vec
    return {t: _embedding_cache[t] for t in titles if t in _embedding_cache}


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))   # already L2-normalized


def _rank_frontier(candidates: set[str], target_title: str, beam_width: int) -> list[str]:
    all_titles = list(candidates) + [target_title]
    embeddings = _embed(all_titles)
    if not embeddings or target_title not in embeddings:
        return list(candidates)[:beam_width]
    target_vec = embeddings[target_title]
    scored = [
        (title, _cosine(embeddings[title], target_vec) if title in embeddings else 0.0)
        for title in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:beam_width]]


# ------------------------------------------------------------------
# Main search
# ------------------------------------------------------------------

async def semantic_bidirectional_search(
    source: GraphSource,
    start: str,
    end: str,
    trace_file: Optional[str] = None,
    event_queue: Optional[asyncio.Queue] = None,
) -> Optional[SearchResult]:
    """Bidirectional best-first search with semantic guidance."""
    if start == end:
        return SearchResult(path=[start], nodes_explored=0, elapsed=0.0, met_at=start)

    t0 = time.perf_counter()
    nodes_explored = 0

    fwd_parents: dict[str, Optional[str]] = {start: None}
    bwd_parents: dict[str, Optional[str]] = {end: None}
    fwd_frontier: set[str] = {start}
    bwd_frontier: set[str] = {end}
    trace: list[dict] = []

    def _check_intersection() -> Optional[str]:
        overlap = set(fwd_parents) & set(bwd_parents)
        if overlap:
            return min(overlap, key=lambda n: _path_len(n, fwd_parents) + _path_len(n, bwd_parents))
        return None

    def _path_len(node: str, parents: dict) -> int:
        n, count = node, 0
        while parents.get(n) is not None:
            n = parents[n]
            count += 1
        return count

    for depth in range(CONFIG.max_depth):
        if not fwd_frontier or not bwd_frontier:
            break
        if nodes_explored >= CONFIG.node_budget:
            if event_queue:
                await event_queue.put({"type": "status", "message": f"Node budget ({CONFIG.node_budget}) reached"})
            break

        expand_fwd = len(fwd_frontier) <= len(bwd_frontier)

        if expand_fwd:
            to_expand = _rank_frontier(fwd_frontier, end, CONFIG.beam_width)
            if trace_file:
                trace.append({"step": depth, "direction": "forward_semantic", "expanding": to_expand})
            if event_queue:
                await event_queue.put({"type": "status", "message": f"Semantic forward expansion — depth {depth+1} (top {len(to_expand)} of {len(fwd_frontier)})…"})
            expansions = await _expand_frontier(source, to_expand, "forward", event_queue)
            new_nodes = 0
            for parent, children in expansions.items():
                for child in children:
                    if child not in fwd_parents:
                        fwd_parents[child] = parent
                        fwd_frontier.add(child)
                        new_nodes += 1
            fwd_frontier -= set(to_expand)
            nodes_explored += new_nodes
        else:
            to_expand = _rank_frontier(bwd_frontier, start, CONFIG.beam_width)
            if trace_file:
                trace.append({"step": depth, "direction": "backward_semantic", "expanding": to_expand})
            if event_queue:
                await event_queue.put({"type": "status", "message": f"Semantic backward expansion — depth {depth+1} (top {len(to_expand)} of {len(bwd_frontier)})…"})
            expansions = await _expand_frontier(source, to_expand, "backward", event_queue)
            new_nodes = 0
            for parent, children in expansions.items():
                for child in children:
                    if child not in bwd_parents:
                        bwd_parents[child] = parent
                        bwd_frontier.add(child)
                        new_nodes += 1
            bwd_frontier -= set(to_expand)
            nodes_explored += new_nodes

        if event_queue:
            await event_queue.put({"type": "stats", "nodes_explored": nodes_explored, "elapsed": round(time.perf_counter() - t0, 2)})

        meeting = _check_intersection()
        if meeting:
            path = _reconstruct_path(meeting, fwd_parents, bwd_parents)
            elapsed = time.perf_counter() - t0
            if trace_file:
                with open(trace_file, "w") as f:
                    for entry in trace:
                        f.write(json.dumps(entry) + "\n")
            return SearchResult(path=path, nodes_explored=nodes_explored, elapsed=elapsed, met_at=meeting)

    elapsed = time.perf_counter() - t0
    if trace_file:
        with open(trace_file, "w") as f:
            for entry in trace:
                f.write(json.dumps(entry) + "\n")
    return None
