"""Observation enrichment protocol bridging perception producers and runtime pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class EnrichmentRequest:
    """Formal observation enrichment request passed from ``ObservationPipeline`` to enrichers."""

    observation: dict[str, Any]
    correlation: dict[str, Any] = field(default_factory=dict)
    audience: str = "actor"
    reply: str = "observe"
    before_observation_id: str | None = None
    suppress_overlay: bool = False
    basis_valid: bool = True


@dataclass(frozen=True)
class Enrichment:
    """Model-facing content parts plus structured sideband facts produced by an enricher."""

    parts: tuple[dict[str, Any], ...] = ()
    facts: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ObservationEnricher(Protocol):
    """Sideband observation enricher invoked by ``ObservationPipeline`` without upward imports."""

    def enrich(self, host: Any, request: EnrichmentRequest) -> Enrichment:
        """Return content ``parts`` and typed ``facts`` derived from ``request.observation``."""
        ...
