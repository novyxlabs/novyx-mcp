"""
Cloud backend for Novyx MCP.

Thin wrapper around the Novyx Python SDK client.
Preserves exact existing behavior — every method delegates to the SDK.
"""

from __future__ import annotations

import os
from typing import Any

from novyx import Novyx


class CloudBackend:
    """Delegates all operations to the Novyx Cloud API via the Python SDK."""

    def __init__(self) -> None:
        api_key = os.environ.get("NOVYX_API_KEY")
        if not api_key:
            raise ValueError("NOVYX_API_KEY environment variable is required")
        try:
            self._client = Novyx(api_key=api_key, source="mcp")
        except TypeError:
            # Older SDK versions don't accept source=
            self._client = Novyx(api_key=api_key)

    # ------------------------------------------------------------------
    # Core Memory
    # ------------------------------------------------------------------

    def remember(self, observation: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.remember(observation, **kwargs)

    def recall(self, query: str, **kwargs: Any) -> Any:
        """Returns the SDK's SearchResult object."""
        return self._client.recall(query, **kwargs)

    def forget(self, memory_id: str) -> bool:
        return self._client.forget(memory_id)

    def memories(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._client.memories(**kwargs)

    def memory(self, memory_id: str) -> dict[str, Any]:
        return self._client.memory(memory_id)

    def stats(self) -> dict[str, Any]:
        return self._client.stats()

    # ------------------------------------------------------------------
    # Rollback & Audit
    # ------------------------------------------------------------------

    def rollback(self, target: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.rollback(target, **kwargs)

    def audit(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._client.audit(**kwargs)

    # ------------------------------------------------------------------
    # Links & Knowledge Graph
    # ------------------------------------------------------------------

    def link(self, source_id: str, target_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.link(source_id, target_id, **kwargs)

    def triple(self, subject: str, predicate: str, object_name: str) -> dict[str, Any]:
        return self._client.triple(subject, predicate, object_name)

    def triples(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.triples(**kwargs)

    # ------------------------------------------------------------------
    # Context Spaces
    # ------------------------------------------------------------------

    def create_space(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.create_space(name=name, **kwargs)

    def list_spaces(self) -> dict[str, Any]:
        return self._client.list_spaces()

    def get_space(self, space_id: str) -> dict[str, Any]:
        return self._client.get_space(space_id)

    def update_space(self, space_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.update_space(space_id, **kwargs)

    def delete_space(self, space_id: str) -> None:
        self._client.delete_space(space_id)

    def space_memories(self, space_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.space_memories(space_id, **kwargs)

    def share_context(self, tag: str, to_email: str, permission: str = "read") -> dict[str, Any]:
        return self._client.share_context(tag, to_email=to_email, permission=permission)

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    def usage(self) -> dict[str, Any]:
        return self._client.usage()

    # ------------------------------------------------------------------
    # Replay (Pro+)
    # ------------------------------------------------------------------

    def replay_timeline(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.replay_timeline(**kwargs)

    def replay_snapshot(self, at: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.replay_snapshot(at, **kwargs)

    def replay_lifecycle(self, memory_id: str) -> dict[str, Any]:
        return self._client.replay_lifecycle(memory_id)

    def replay_diff(self, start: str, end: str) -> dict[str, Any]:
        return self._client.replay_diff(start, end)

    # ------------------------------------------------------------------
    # Cortex (Pro+)
    # ------------------------------------------------------------------

    def cortex_status(self) -> dict[str, Any]:
        return self._client.cortex_status()

    def cortex_run(self) -> dict[str, Any]:
        return self._client.cortex_run()

    def cortex_insights(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.cortex_insights(**kwargs)
