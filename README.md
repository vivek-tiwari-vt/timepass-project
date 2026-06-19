# ConceptBridge

Connect any two ideas through the Wikipedia link graph — and explain how they're related.

## What it does

ConceptBridge takes two free-text concepts (e.g. "Jazz" and "Quantum computing"), finds the
shortest chain of Wikipedia hyperlinks connecting them, and explains each hop in plain English.

It generalises the *Wikiracing* game into a tool for exploring conceptual relationships.

## Why Wikipedia

Wikipedia is the substrate because:
- Concept → page resolution is clean (MediaWiki opensearch API)
- The graph is strongly connected — almost everything reaches everything
- **Critically**, the MediaWiki API exposes both outgoing links AND backlinks (incoming links),
  which enables bidirectional BFS — roughly halving search depth

> Note: the links are untyped "related-somehow" edges, not a typed knowledge graph.
> Paths show a chain of association, not strict logical entailment.

## Architecture

```
sources/
  base.py          – GraphSource interface (resolve, outgoing, incoming)
  wikipedia.py     – MediaWiki implementation + OpenWebSource stub
search/
  bfs.py           – Bidirectional BFS (plain)
  semantic_bfs.py  – Bidirectional best-first with sentence-transformer ranking
explain.py         – LLM hop explanations via Anthropic API (claude-sonnet)
api/app.py         – FastAPI backend + graph JSON export
frontend/index.html– Interactive Cytoscape.js graph UI
config.py          – Single config dataclass
main.py            – CLI entrypoint
eval.py            – Eval harness (BFS vs semantic, fixed pairs)
```

## Quick start

```bash
pip install -r requirements.txt

# CLI — find a path (steps 1-2)
python main.py "Jazz" "Quantum computing" --no-explain

# CLI — with LLM explanations via Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...
python main.py "Jazz" "Quantum computing"

# CLI — with LLM explanations via DeepSeek
export DEEPSEEK_API_KEY=sk-...
python main.py "Jazz" "Quantum computing" --provider deepseek

# Switch default provider in config.py: llm_provider = "deepseek"

# Web UI
uvicorn api.app:app --reload
# open http://localhost:8000
```

## Bidirectional search

Forward BFS from start follows *outgoing* links.
Backward BFS from end follows *backlinks* (incoming).
They meet in the middle, cutting depth roughly in half.

### Semantic guidance (step 3)

When `--semantic` is on (default), frontier nodes are ranked by cosine similarity of their
title embedding (all-MiniLM-L6-v2) toward the opposite target. Only the top `beam_width`
nodes are expanded per round. This makes the search faster and produces more meaningful paths
without sacrificing correctness for shallow graphs.

## Open-web extension

`OpenWebSource` in `sources/wikipedia.py` is stubbed as a TODO. The open web version would:
- Resolve concepts via DuckDuckGo or Google Custom Search
- Follow `<a href>` links parsed from fetched pages
- **Not support backlinks** (no public index exposes them for free)
- Degrade to forward-only greedy search — fuzzier and noisier than Wikipedia paths

## Eval

```bash
python eval.py
```

Runs six concept pairs through bidirectional BFS and semantic BFS, printing path length,
nodes explored, and wall-clock for each. Results saved to `eval_results.jsonl`.

## Config

Edit `config.py`:
- `max_depth` / `node_budget` — search limits
- `beam_width` — semantic expansion width
- `skip_hub_pages` — skip year pages, "United States", list pages
- `llm_model` — Anthropic model for hop explanations
- `embedding_model` — sentence-transformers model
