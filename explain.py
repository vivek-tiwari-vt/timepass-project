"""LLM-powered hop-by-hop explanations using Anthropic API."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import anthropic

from config import CONFIG
from sources.wikipedia import WikipediaSource


@dataclass
class HopExplanation:
    from_title: str
    to_title: str
    sentence: str


async def explain_hop(
    client: anthropic.AsyncAnthropic,
    source: WikipediaSource,
    from_title: str,
    to_title: str,
) -> HopExplanation:
    from_intro, to_intro = await asyncio.gather(
        source.get_page_intro(from_title),
        source.get_page_intro(to_title),
    )

    prompt = (
        f'You are explaining a conceptual link in a Wikipedia path.\n\n'
        f'Source article: "{from_title}"\n'
        f'Context: {from_intro[:300]}\n\n'
        f'Target article: "{to_title}"\n'
        f'Context: {to_intro[:300]}\n\n'
        f'Write exactly ONE concise sentence (max 30 words) that explains the conceptual '
        f'bridge from "{from_title}" to "{to_title}". No preamble, just the sentence.'
    )

    msg = await client.messages.create(
        model=CONFIG.llm_model,
        max_tokens=CONFIG.llm_max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    sentence = msg.content[0].text.strip()
    return HopExplanation(from_title=from_title, to_title=to_title, sentence=sentence)


async def explain_path(
    source: WikipediaSource,
    path: list[str],
) -> list[HopExplanation]:
    """Generate a one-sentence bridge explanation for every hop in `path`."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Return placeholder explanations if key is missing
        return [
            HopExplanation(from_title=path[i], to_title=path[i + 1], sentence="[LLM explanation unavailable — set ANTHROPIC_API_KEY]")
            for i in range(len(path) - 1)
        ]

    client = anthropic.AsyncAnthropic(api_key=api_key)
    tasks = [
        explain_hop(client, source, path[i], path[i + 1])
        for i in range(len(path) - 1)
    ]
    return await asyncio.gather(*tasks)


def format_detail_view(path: list[str], explanations: list[HopExplanation], stats: dict) -> str:
    lines = ["", "═" * 60, "  ConceptBridge — Path Detail", "═" * 60, ""]
    for i, title in enumerate(path):
        lines.append(f"  {i}. {title}")
        if i < len(explanations):
            lines.append(f"       └─ {explanations[i].sentence}")
            lines.append("")
    lines.append("─" * 60)
    lines.append(f"  Path length  : {len(path) - 1} hop(s)")
    lines.append(f"  Nodes explored: {stats.get('nodes_explored', '?')}")
    lines.append(f"  Wall-clock   : {stats.get('elapsed', 0):.2f}s")
    lines.append(f"  Met at node  : {stats.get('met_at', '?')}")
    lines.append("═" * 60)
    return "\n".join(lines)
