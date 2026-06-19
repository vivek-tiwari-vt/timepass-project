"""Evaluation harness: compare forward BFS vs bidirectional BFS vs bidirectional+semantic."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from typing import Optional

from config import CONFIG
from search.bfs import bidirectional_bfs
from search.semantic_bfs import semantic_bidirectional_search
from sources.wikipedia import WikipediaSource

EVAL_PAIRS = [
    ("Jazz", "Quantum computing"),
    ("Photosynthesis", "Internet"),
    ("Ancient Rome", "Machine learning"),
    ("Beethoven", "Black hole"),
    ("Chess", "DNA"),
    ("Philosophy", "Nuclear fusion"),
]


@dataclass
class EvalResult:
    pair: tuple
    method: str
    success: bool
    path_length: int
    nodes_explored: int
    elapsed: float
    path: list


async def run_pair(source: WikipediaSource, start: str, end: str, method: str) -> EvalResult:
    t0 = time.perf_counter()
    try:
        start_resolved = (await source.resolve_concept(start)).title
        end_resolved = (await source.resolve_concept(end)).title

        if method == "bidirectional_bfs":
            result = await bidirectional_bfs(source, start_resolved, end_resolved)
        elif method == "semantic":
            result = await semantic_bidirectional_search(source, start_resolved, end_resolved)
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.perf_counter() - t0
        if result:
            return EvalResult(
                pair=(start, end),
                method=method,
                success=True,
                path_length=len(result.path) - 1,
                nodes_explored=result.nodes_explored,
                elapsed=elapsed,
                path=result.path,
            )
    except Exception as e:
        print(f"[eval] Error ({method}, {start}→{end}): {e}")

    return EvalResult(
        pair=(start, end),
        method=method,
        success=False,
        path_length=-1,
        nodes_explored=0,
        elapsed=time.perf_counter() - t0,
        path=[],
    )


async def main():
    source = WikipediaSource()
    all_results: list[EvalResult] = []

    methods = ["bidirectional_bfs", "semantic"]
    for start, end in EVAL_PAIRS:
        print(f"\n{'='*60}")
        print(f"Pair: {start} → {end}")
        for method in methods:
            print(f"  Running {method}…")
            r = await run_pair(source, start, end, method)
            all_results.append(r)
            status = "✓" if r.success else "✗"
            print(f"  {status} {method}: path={r.path_length} hops, explored={r.nodes_explored}, time={r.elapsed:.2f}s")
            if r.success:
                print(f"    Path: {' → '.join(r.path)}")

    await source.close()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for method in methods:
        results = [r for r in all_results if r.method == method]
        successes = [r for r in results if r.success]
        print(f"\n{method}:")
        print(f"  Success rate: {len(successes)}/{len(results)}")
        if successes:
            avg_path = sum(r.path_length for r in successes) / len(successes)
            avg_nodes = sum(r.nodes_explored for r in successes) / len(successes)
            avg_time = sum(r.elapsed for r in successes) / len(successes)
            print(f"  Avg path length : {avg_path:.1f}")
            print(f"  Avg nodes explored: {avg_nodes:.0f}")
            print(f"  Avg time        : {avg_time:.2f}s")

    # Save full results
    with open("eval_results.jsonl", "w") as f:
        for r in all_results:
            d = asdict(r)
            d["pair"] = list(d["pair"])
            f.write(json.dumps(d) + "\n")
    print("\nResults saved to eval_results.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
