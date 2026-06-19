"""GraphSource interface — all link-graph backends implement this."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ResolvedConcept:
    title: str           # canonical page/document title
    url: str             # canonical URL
    alternatives: list   # other candidate titles from search


class GraphSource(ABC):
    @abstractmethod
    async def resolve_concept(self, concept: str) -> ResolvedConcept:
        """Resolve a free-text concept to a canonical node."""

    @abstractmethod
    async def get_outgoing_links(self, title: str) -> list[str]:
        """Return titles of pages linked FROM `title`."""

    @abstractmethod
    async def get_incoming_links(self, title: str) -> list[str]:
        """Return titles of pages that link TO `title` (backlinks)."""
