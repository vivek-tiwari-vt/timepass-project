"""ConceptBridge configuration."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # Wikipedia / MediaWiki
    wikipedia_user_agent: str = "ConceptBridge/1.0 (vivek.bt.tiwari@gmail.com)"
    wikipedia_api_url: str = "https://en.wikipedia.org/w/api.php"
    wikipedia_batch_size: int = 50          # links per API batch call
    max_links_per_page: int = 200           # cap links fetched per page (None = unlimited)

    # Search / BFS
    max_depth: int = 6                      # max hops each direction
    node_budget: int = 5000                 # total nodes before giving up
    beam_width: int = 10                    # best-first frontier expansion width
    skip_hub_pages: bool = True             # skip year pages, disambiguation, list pages

    # Semantic ranking
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_cache_size: int = 10_000

    # LLM — provider: "anthropic" | "deepseek"
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 200

    # DeepSeek (OpenAI-compatible)
    deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions"
    deepseek_model: str = "deepseek-chat"

    # Hub-skip patterns (regexes matched against page title)
    hub_skip_patterns: list = field(default_factory=lambda: [
        r"^\d{4}$",                         # year pages
        r"^List of",
        r"^Lists of",
        r"^Index of",
        r"^Outline of",
        r"United States$",
        r"^Wikipedia:",
        r"^Template:",
        r"^Category:",
        r"^File:",
        r"^Help:",
    ])

    # Output
    trace_file: str = "run_trace.jsonl"


CONFIG = Config()
