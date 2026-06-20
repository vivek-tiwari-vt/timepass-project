"""Bidirectional BFS over a GraphSource.

Forward frontier: BFS from start following outgoing links.
Backward frontier: BFS from end following incoming (back)links.
Meet in the middle -> reconstruct path.

Pass an asyncio.Queue as `event_queue` to receive live progress events
(used by the SSE streaming endpoint).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from config import CONFIG
from sources.base import GraphSource


@dataclass
class SearchResult:
    path: list[str]                 # ordered: start -> ... -> end
    nodes_explored: int
    elapsed: float                  # seconds
    met_at: str                     # the node where the two frontiers met


def _is_hub(title: str) -> bool:
    if not CONFIG.skip_hub_pages:
        return False
    for pattern in CONFIG.hub_skip_patterns:
        if re.search(pattern, title):
            return True
    return False


async def _expand_frontier(
    source: GraphSource,
    titles: list[str],
    direction: str,             # "forward" | "backward"
    event_queue: Optional[asyncio.Queue] = None,
) -> dict[str, list[str]]:
    """Fetch links for a batch of titles in parallel, emitting events per node."""

    async def fetch_one(title: str) -> tuple[str, list[str]]:
        if direction == "forward":
            links = await source.get_outgoing_links(title)
        else:
            links = await source.get_incoming_links(title)
        filtered = [l for l in links if not _is_hub(l)]
        if event_queue is not None:
            await event_queue.put({
                "type": "expand_batch",
                "direction": direction,
                "parent": title,
                "children": filtered[:40],      # cap payload for browser
                "total_links": len(filtered),
            })
        return title, filtered

    tasks = [fetch_one(t) for t in titles]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: dict[str, list[str]] = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        title, links = r
        out[title] = links
    return out


def _reconstruct_path(
    meeting_node: str,
    fwd_parents: dict[str, Optional[str]],
    bwd_parents: dict[str, Optional[str]],
) -> list[str]:
    fwd_path: list[str] = []
    node: Optional[str] = meeting_node
    while node is not None:
        fwd_path.append(node)
        node = fwd_parents.get(node)
    fwd_path.reverse()

    bwd_path: list[str] = []
    node = bwd_parents.get(meeting_node)
    while node is not None:
        bwd_path.append(node)
        node = bwd_parents.get(node)

    return fwd_path + bwd_path


async def bidirectional_bfs(
    source: GraphSource,
    start: str,
    end: str,
    trace_file: Optional[str] = None,
    event_queue: Optional[asyncio.Queue] = None,
) -> Optional[SearchResult]:
    """Run bidirectional BFS and return the first connecting path found."""
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

    step = 0
    for depth in range(CONFIG.max_depth):
        if not fwd_frontier or not bwd_frontier:
            break
        if nodes_explored >= CONFIG.node_budget:
            if event_queue:
                await event_queue.put({"type": "status", "message": f"Node budget ({CONFIG.node_budget}) reached"})
            break

        if len(fwd_frontier) <= len(bwd_frontier):
            step += 1
            if trace_file:
                trace.append({"step": step, "direction": "forward", "frontier": list(fwd_frontier)})
            if event_queue:
                await event_queue.put({"type": "status", "message": f"Expanding forward frontier — depth {depth+1} ({len(fwd_frontier)} nodes)…"})
            expansions = await _expand_frontier(source, list(fwd_frontier), "forward", event_queue)
            new_frontier: set[str] = set()
            for parent, children in expansions.items():
                for child in children:
                    if child not in fwd_parents:
                        fwd_parents[child] = parent
                        new_frontier.add(child)
                        nodes_explored += 1
            fwd_frontier = new_frontier
        else:
            step += 1
            if trace_file:
                trace.append({"step": step, "direction": "backward", "frontier": list(bwd_frontier)})
            if event_queue:
                await event_queue.put({"type": "status", "message": f"Expanding backward frontier — depth {depth+1} ({len(bwd_frontier)} nodes)…"})
            expansions = await _expand_frontier(source, list(bwd_frontier), "backward", event_queue)
            new_frontier = set()
            for parent, children in expansions.items():
                for child in children:
                    if child not in bwd_parents:
                        bwd_parents[child] = parent
                        new_frontier.add(child)
                        nodes_explored += 1
            bwd_frontier = new_frontier

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

    if trace_file:
        with open(trace_file, "w") as f:
            for entry in trace:
                f.write(json.dumps(entry) + "\n")
    return None
