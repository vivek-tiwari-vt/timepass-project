"""ConceptBridge CLI — steps 1-2: resolve concepts + bidirectional BFS."""

from __future__ import annotations

import argparse
import asyncio
import sys

from config import CONFIG
from explain import explain_path, format_detail_view
from search.bfs import bidirectional_bfs
from search.semantic_bfs import semantic_bidirectional_search
from sources.wikipedia import WikipediaSource


async def run(start_text: str, end_text: str, semantic: bool, no_explain: bool):
    source = WikipediaSource()
    try:
        print(f"\nResolving concepts…")
        start_concept, end_concept = await asyncio.gather(
            source.resolve_concept(start_text),
            source.resolve_concept(end_text),
        )
        print(f"  Start → '{start_concept.title}'", end="")
        if start_concept.alternatives:
            print(f"  (alternatives: {', '.join(start_concept.alternatives[:3])})", end="")
        print()
        print(f"  End   → '{end_concept.title}'", end="")
        if end_concept.alternatives:
            print(f"  (alternatives: {', '.join(end_concept.alternatives[:3])})", end="")
        print()

        print(f"\nSearching {'(semantic)' if semantic else '(plain BFS)'}…")
        search_fn = semantic_bidirectional_search if semantic else bidirectional_bfs
        result = await search_fn(
            source,
            start_concept.title,
            end_concept.title,
            trace_file=CONFIG.trace_file,
        )

        if result is None:
            print("\n[!] No path found within budget. Try increasing max_depth or node_budget.")
            return

        print(f"\nPath found ({len(result.path)-1} hop(s)):")
        print("  " + " → ".join(result.path))

        if not no_explain:
            print("\nGenerating hop explanations…")
            explanations = await explain_path(source, result.path)
            stats = {
                "nodes_explored": result.nodes_explored,
                "elapsed": result.elapsed,
                "met_at": result.met_at,
            }
            print(format_detail_view(result.path, explanations, stats))
        else:
            print(f"\nStats: {result.nodes_explored} nodes explored, {result.elapsed:.2f}s")

    finally:
        await source.close()


def main():
    parser = argparse.ArgumentParser(description="ConceptBridge: connect any two concepts via Wikipedia")
    parser.add_argument("start", help='Starting concept, e.g. "Jazz"')
    parser.add_argument("end", help='Ending concept, e.g. "Quantum computing"')
    parser.add_argument("--plain-bfs", action="store_true", help="Use plain BFS without semantic guidance")
    parser.add_argument("--no-explain", action="store_true", help="Skip LLM hop explanations")
    args = parser.parse_args()

    asyncio.run(run(args.start, args.end, not args.plain_bfs, args.no_explain))


if __name__ == "__main__":
    main()
