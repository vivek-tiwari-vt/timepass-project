"""Bidirectional BFS over a GraphSource.

Forward frontier: BFS from start following outgoing links.
Backward frontier: BFS from end following incoming (back)links.
Meet in the middle -> reconstruct path.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
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
    direction: str,   # "forward" | "backward"
) -> dict[str, list[str]]:
    """Fetch links for a batch of titles in parallel."""
    async def fetch_one(title: str) -> tuple[str, list[str]]:
        if direction == "forward":
            links = await source.get_outgoing_links(title)
        else:
            links = await source.get_incoming_links(title)
        return title, [l for l in links if not _is_hub(l)]

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
    # Forward path: start -> ... -> meeting_node
    fwd_path: list[str] = []
    node: Optional[str] = meeting_node
    while node is not None:
        fwd_path.append(node)
        node = fwd_parents.get(node)
    fwd_path.reverse()

    # Backward path: meeting_node -> ... -> end
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
) -> Optional[SearchResult]:
    """Run bidirectional BFS and return the first connecting path found."""
    if start == end:
        return SearchResult(path=[start], nodes_explored=0, elapsed=0.0, met_at=start)

    t0 = time.perf_counter()
    nodes_explored = 0

    # parent maps: node -> its parent in BFS tree (None for roots)
    fwd_parents: dict[str, Optional[str]] = {start: None}
    bwd_parents: dict[str, Optional[str]] = {end: None}

    # BFS queues hold the *current depth frontier* as sets
    fwd_frontier: set[str] = {start}
    bwd_frontier: set[str] = {end}

    trace: list[dict] = []

    def _trace(step: int, direction: str, frontier: list[str]):
        if trace_file:
            trace.append({"step": step, "direction": direction, "frontier": frontier})

    def _check_intersection() -> Optional[str]:
        overlap = set(fwd_parents) & set(bwd_parents)
        if overlap:
            # pick the node whose total path is shortest
            return min(overlap, key=lambda n: (
                _path_len(n, fwd_parents) + _path_len(n, bwd_parents)
            ))
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
            print(f"[bfs] node budget ({CONFIG.node_budget}) exceeded")
            break

        # Expand whichever frontier is smaller (minimizes work)
        if len(fwd_frontier) <= len(bwd_frontier):
            # --- expand forward ---
            step += 1
            _trace(step, "forward", list(fwd_frontier))
            expansions = await _expand_frontier(source, list(fwd_frontier), "forward")
            new_frontier: set[str] = set()
            for parent, children in expansions.items():
                for child in children:
                    if child not in fwd_parents:
                        fwd_parents[child] = parent
                        new_frontier.add(child)
                        nodes_explored += 1
            fwd_frontier = new_frontier
            print(f"[bfs] fwd depth {depth+1}: frontier={len(fwd_frontier)}, explored={nodes_explored}")
        else:
            # --- expand backward ---
            step += 1
            _trace(step, "backward", list(bwd_frontier))
            expansions = await _expand_frontier(source, list(bwd_frontier), "backward")
            new_frontier = set()
            for parent, children in expansions.items():
                for child in children:
                    if child not in bwd_parents:
                        bwd_parents[child] = parent
                        new_frontier.add(child)
                        nodes_explored += 1
            bwd_frontier = new_frontier
            print(f"[bfs] bwd depth {depth+1}: frontier={len(bwd_frontier)}, explored={nodes_explored}")

        meeting = _check_intersection()
        if meeting:
            path = _reconstruct_path(meeting, fwd_parents, bwd_parents)
            elapsed = time.perf_counter() - t0
            if trace_file:
                with open(trace_file, "w") as f:
                    for entry in trace:
                        f.write(json.dumps(entry) + "\n")
            return SearchResult(
                path=path,
                nodes_explored=nodes_explored,
                elapsed=elapsed,
                met_at=meeting,
            )

    elapsed = time.perf_counter() - t0
    if trace_file:
        with open(trace_file, "w") as f:
            for entry in trace:
                f.write(json.dumps(entry) + "\n")
    return None
