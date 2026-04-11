"""
Novyx MCP Server

Exposes Novyx memory operations as MCP tools, resources, and prompts
for Claude Desktop, Cursor, and Claude Code.

Works in two modes:
- Cloud mode: Set NOVYX_API_KEY for full Novyx Cloud features
- Local mode: No API key needed — memories stored in ~/.novyx/local.db
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

try:
    from mcp.types import ToolAnnotations
except ImportError:
    # Fallback for environments where mcp.types is unavailable
    from dataclasses import dataclass

    @dataclass
    class ToolAnnotations:  # type: ignore[no-redef]
        readOnlyHint: bool | None = None
        destructiveHint: bool | None = None
        idempotentHint: bool | None = None


mcp = FastMCP("novyx-memory")

# Singleton backend instance
_backend_instance = None


def _get_backend() -> Any:
    """Get the memory backend — cloud (Novyx API) or local (SQLite).

    Cloud mode: NOVYX_API_KEY is set → delegates to Novyx Cloud API
    Local mode: No API key → full offline operation via SQLite
    """
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    if os.environ.get("NOVYX_API_KEY"):
        from .cloud_backend import CloudBackend

        _backend_instance = CloudBackend()
    else:
        from .local_backend import LocalBackend

        _backend_instance = LocalBackend()

    return _backend_instance


# Cached git repo name (None = not yet detected, "" = detection failed)
_git_repo_name: str | None = None


def _detect_git_repo() -> str:
    """Detect the current git repository name. Cached per session.

    Returns the repo directory name (e.g. 'novyx-core') or empty string if
    not in a git repo or git is unavailable.
    """
    global _git_repo_name
    if _git_repo_name is not None:
        return _git_repo_name
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            _git_repo_name = os.path.basename(result.stdout.strip())
        else:
            _git_repo_name = ""
    except Exception:
        _git_repo_name = ""
    return _git_repo_name


def _inject_repo_tag(tags: list[str] | None) -> list[str] | None:
    """Add a repo:<name> tag if we're inside a git repository.

    Returns the original tags (possibly with the repo tag appended), or None
    if no tags and no repo detected.
    """
    repo = _detect_git_repo()
    if not repo:
        return tags
    repo_tag = f"repo:{repo}"
    if tags is None:
        return [repo_tag]
    if repo_tag not in tags:
        return tags + [repo_tag]
    return tags


def _handle_tier_error(e: Exception, feature: str = "This feature") -> str:
    """Return a friendly upgrade message for tier-gated or cloud-only features."""
    from .local_backend import CloudFeatureError

    if isinstance(e, CloudFeatureError):
        return json.dumps(
            {
                "error": str(e),
                "upgrade": "https://novyxlabs.com/pricing",
            }
        )
    error_str = str(e)
    if "403" in error_str or "feature" in error_str.lower() or "Forbidden" in error_str:
        return json.dumps(
            {
                "error": f"{feature} requires Pro tier or higher.",
                "upgrade": "https://novyxlabs.com/pricing",
            }
        )
    return json.dumps({"error": error_str})


def _json_error(e: Exception) -> str:
    """Serialize a plain error response."""
    return json.dumps({"error": str(e)})


def _json_result(result: Any) -> str:
    """Serialize a successful MCP tool result."""
    return json.dumps(result, default=str)


def _call_backend_json(method_name: str, *args: Any, **kwargs: Any) -> str:
    """Call a backend method and serialize the result as JSON."""
    try:
        backend = _get_backend()
        method = getattr(backend, method_name)
        return _json_result(method(*args, **kwargs))
    except Exception as e:
        return _json_error(e)


# =========================================================================
# Tools
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def remember(
    observation: str,
    tags: list[str] | None = None,
    importance: int = 5,
    context: str | None = None,
    ttl_seconds: int | None = None,
) -> str:
    """Store a memory observation in Novyx.

    Args:
        observation: The memory content to store.
        tags: Optional list of tags for categorization.
        importance: Importance score 1-10 (default 5).
        context: Optional context string.
        ttl_seconds: Optional time-to-live in seconds. Memory auto-expires after this duration.

    Returns:
        JSON string with the stored memory UUID and details.
    """
    try:
        backend = _get_backend()
        enriched_tags = _inject_repo_tag(tags)
        kwargs: dict[str, Any] = {
            "observation": observation,
            "importance": importance,
        }
        if enriched_tags is not None:
            kwargs["tags"] = enriched_tags
        if context is not None:
            kwargs["context"] = context
        if ttl_seconds is not None:
            kwargs["ttl_seconds"] = ttl_seconds
        result = backend.remember(**kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def draft_memory(
    observation: str,
    tags: list[str] | None = None,
    importance: int = 5,
    context: str | None = None,
    confidence: float = 1.0,
    branch_id: str | None = None,
) -> str:
    """Create a reviewable draft without writing to canonical memory.

    Args:
        observation: Proposed memory content.
        tags: Optional list of tags for categorization.
        importance: Importance score 1-10 (default 5).
        context: Optional context string.
        confidence: Confidence score 0-1 (default 1.0).

    Returns:
        JSON string with the draft, review summary, and similar memories.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {
            "observation": observation,
            "importance": importance,
            "confidence": confidence,
        }
        if tags is not None:
            kwargs["tags"] = tags
        if context is not None:
            kwargs["context"] = context
        if branch_id is not None:
            kwargs["branch_id"] = branch_id
        result = backend.draft_memory(**kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def memory_drafts(status: str | None = None, branch_id: str | None = None) -> str:
    """List current memory drafts.

    Args:
        status: Optional filter by status (draft, merged, rejected).

    Returns:
        JSON string with draft records.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {}
        if status is not None:
            kwargs["status"] = status
        if branch_id is not None:
            kwargs["branch_id"] = branch_id
        result = backend.memory_drafts(**kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def draft_diff(draft_id: str, compare_to: str | None = None) -> str:
    """Show a field-level diff for a memory draft.

    Args:
        draft_id: Draft identifier returned by draft_memory.
        compare_to: Optional existing memory UUID to compare against.

    Returns:
        JSON string with changed fields and a merge recommendation.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {}
        if compare_to is not None:
            kwargs["compare_to"] = compare_to
        result = backend.draft_diff(draft_id, **kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def merge_draft(draft_id: str, supersede_memory_id: str | None = None) -> str:
    """Merge a reviewed draft into canonical memory.

    Args:
        draft_id: Draft identifier returned by draft_memory.
        supersede_memory_id: Optional older memory to mark as superseded.

    Returns:
        JSON string describing the merged memory.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {}
        if supersede_memory_id is not None:
            kwargs["supersede_memory_id"] = supersede_memory_id
        result = backend.merge_draft(draft_id, **kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def memory_branch(branch_id: str) -> str:
    """Get grouped review information for a branch/session of drafts."""
    try:
        backend = _get_backend()
        result = backend.memory_branch(branch_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def merge_branch(branch_id: str) -> str:
    """Merge all open drafts in a branch/session."""
    try:
        backend = _get_backend()
        result = backend.merge_branch(branch_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def reject_draft(draft_id: str, reason: str | None = None) -> str:
    """Reject a draft without creating a memory.

    Args:
        draft_id: Draft identifier returned by draft_memory.
        reason: Optional reason for rejection.

    Returns:
        JSON string with the rejected draft state.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {}
        if reason is not None:
            kwargs["reason"] = reason
        result = backend.reject_draft(draft_id, **kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def reject_branch(branch_id: str, reason: str | None = None) -> str:
    """Reject all open drafts in a branch/session."""
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {}
        if reason is not None:
            kwargs["reason"] = reason
        result = backend.reject_branch(branch_id, **kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def recall(
    query: str,
    limit: int = 5,
    tags: list[str] | None = None,
    min_score: float = 0.0,
    explain: bool = False,
) -> str:
    """Search memories semantically using natural language.

    Args:
        query: Natural language search query.
        limit: Maximum number of results to return (default 5).
        tags: Optional tag filter.
        min_score: Minimum similarity score 0-1 (default 0).
        explain: If true, include scoring breakdown for each result showing why it scored the way it did.

    Returns:
        JSON string with matching memories and their scores.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {"query": query, "limit": limit, "min_score": min_score}
        if tags is not None:
            kwargs["tags"] = tags
        result = backend.recall(**kwargs)

        # Cloud backend returns a SearchResult object; local returns a dict
        if isinstance(result, dict):
            if explain:
                for mem in result.get("memories", []):
                    cosine_score = mem.get("similarity", 0.0) or 0.0
                    imp = mem.get("importance", 5)
                    conf = mem.get("confidence", 1.0)
                    importance_boost = 0.2 * (imp / 10.0)
                    confidence_boost = 0.1 * conf
                    final_score = mem.get("score", 0.0) or 0.0
                    parts = [f"Matched on semantic similarity ({cosine_score:.2f})"]
                    if imp >= 7:
                        parts.append(f"boosted by high importance ({imp}/10)")
                    elif imp <= 3:
                        parts.append(f"low importance ({imp}/10)")
                    if conf < 0.5:
                        parts.append(f"low confidence ({conf:.1f})")
                    mem["scoring_breakdown"] = {
                        "cosine_score": round(cosine_score, 4),
                        "importance_boost": round(importance_boost, 4),
                        "confidence_boost": round(confidence_boost, 4),
                        "final_score": round(final_score, 4),
                        "explanation": ", ".join(parts),
                    }
            return json.dumps(result, default=str)

        # Cloud mode: convert SearchResult to dict
        memories = []
        for mem in result.memories:
            entry: dict[str, Any] = {
                "uuid": mem.uuid,
                "observation": mem.observation,
                "tags": mem.tags,
                "importance": mem.importance,
                "score": mem.score,
                "created_at": mem.created_at,
            }
            if explain:
                cosine_score = getattr(mem, "similarity", None) or 0.0
                imp = mem.importance or 5
                conf = getattr(mem, "confidence", 1.0) or 1.0
                importance_boost = 0.2 * (imp / 10.0)
                confidence_boost = 0.1 * conf
                final_score = mem.score or 0.0
                parts = [f"Matched on semantic similarity ({cosine_score:.2f})"]
                if imp >= 7:
                    parts.append(f"boosted by high importance ({imp}/10)")
                elif imp <= 3:
                    parts.append(f"low importance ({imp}/10)")
                if conf < 0.5:
                    parts.append(f"low confidence ({conf:.1f})")
                entry["scoring_breakdown"] = {
                    "cosine_score": round(cosine_score, 4),
                    "importance_boost": round(importance_boost, 4),
                    "confidence_boost": round(confidence_boost, 4),
                    "final_score": round(final_score, 4),
                    "explanation": ", ".join(parts),
                }
            memories.append(entry)
        return json.dumps(
            {
                "query": result.query,
                "total_results": result.total_results,
                "memories": memories,
            },
            default=str,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def forget(memory_id: str) -> str:
    """Delete a memory by its UUID.

    Args:
        memory_id: The UUID of the memory to delete.

    Returns:
        JSON string indicating success or failure.
    """
    try:
        backend = _get_backend()
        result = backend.forget(memory_id)
        return json.dumps({"success": result, "memory_id": memory_id})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_memories(
    limit: int = 50,
    tags: list[str] | None = None,
) -> str:
    """List stored memories with optional tag filtering.

    Args:
        limit: Maximum number of memories to return (default 50).
        tags: Optional tag filter to narrow results.

    Returns:
        JSON string with the list of memories.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {"limit": limit}
        if tags is not None:
            kwargs["tags"] = tags
        result = backend.memories(**kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def memory_stats() -> str:
    """Get memory statistics for the current account.

    Returns:
        JSON string with total memories, average importance, tag distribution, etc.
    """
    try:
        backend = _get_backend()
        result = backend.stats()
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def rollback(
    target: str,
    dry_run: bool = False,
) -> str:
    """Rollback memory to a point in time.

    Supports ISO timestamps (e.g. '2026-01-15T10:00:00Z') and relative time
    expressions (e.g. '2 hours ago').

    Args:
        target: ISO timestamp or relative time expression.
        dry_run: If true, preview changes without applying them.

    Returns:
        JSON string with rollback results or preview.
    """
    try:
        backend = _get_backend()
        result = backend.rollback(target, dry_run=dry_run)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def audit(
    limit: int = 20,
    operation: str | None = None,
) -> str:
    """Get the audit trail of memory operations.

    Args:
        limit: Maximum number of audit entries to return (default 20).
        operation: Optional filter by operation type (CREATE, UPDATE, DELETE, ROLLBACK).

    Returns:
        JSON string with audit trail entries.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {"limit": limit}
        if operation is not None:
            kwargs["operation"] = operation
        result = backend.audit(**kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def link_memories(
    source_id: str,
    target_id: str,
    relation: str = "related",
) -> str:
    """Create a directed link between two memories.

    Args:
        source_id: UUID of the source memory.
        target_id: UUID of the target memory.
        relation: Type of relationship (default 'related').

    Returns:
        JSON string with the created link details.
    """
    try:
        backend = _get_backend()
        result = backend.link(source_id, target_id, relation=relation)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def add_triple(
    subject: str,
    predicate: str,
    object_name: str,
) -> str:
    """Add a knowledge graph triple (subject -> predicate -> object).

    Entities are auto-created by name if they don't exist.

    Args:
        subject: The subject entity name.
        predicate: The relationship predicate.
        object_name: The object entity name.

    Returns:
        JSON string with the created triple details.
    """
    try:
        backend = _get_backend()
        result = backend.triple(subject, predicate, object_name)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def query_triples(
    subject: str | None = None,
    predicate: str | None = None,
    object_name: str | None = None,
) -> str:
    """Query knowledge graph triples with optional filters.

    At least one filter should be provided. Returns all matching triples.

    Args:
        subject: Filter by subject entity name.
        predicate: Filter by relationship predicate.
        object_name: Filter by object entity name.

    Returns:
        JSON string with matching triples.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {}
        if subject is not None:
            kwargs["subject"] = subject
        if predicate is not None:
            kwargs["predicate"] = predicate
        if object_name is not None:
            kwargs["object"] = object_name
        result = backend.triples(**kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =========================================================================
# Context Spaces — Multi-Agent Collaboration
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def create_space(
    name: str,
    description: str | None = None,
    allowed_agents: list[str] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Create a shared context space for multi-agent collaboration.

    Spaces let multiple agents share memories with fine-grained permissions.
    The creator is the owner and can grant read/write access to other agents or tenants.

    Args:
        name: Name for the space.
        description: Optional description of the space's purpose.
        allowed_agents: Optional list of agent IDs that can access this space.
        tags: Optional tags for the space.

    Returns:
        JSON string with the created space details including space_id.
    """
    try:
        backend = _get_backend()
        result = backend.create_space(
            name=name,
            description=description,
            allowed_agent_ids=allowed_agents,
            tags=tags,
        )
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_spaces() -> str:
    """List all context spaces you can access.

    Returns spaces you own and spaces shared with you.

    Returns:
        JSON string with list of spaces and their memory counts.
    """
    try:
        backend = _get_backend()
        result = backend.list_spaces()
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def space_memories(
    space_id: str,
    query: str | None = None,
    limit: int = 50,
) -> str:
    """List or search memories within a context space.

    Args:
        space_id: The space ID to query.
        query: Optional search query to filter memories semantically.
        limit: Maximum number of memories to return (default 50).

    Returns:
        JSON string with memories in the space.
    """
    try:
        backend = _get_backend()
        result = backend.space_memories(space_id, query=query, limit=limit)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def update_space(
    space_id: str,
    name: str | None = None,
    description: str | None = None,
    allowed_agents: list[str] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Update a context space (owner only).

    Args:
        space_id: The space ID to update.
        name: New name for the space.
        description: New description.
        allowed_agents: Updated list of allowed agent IDs.
        tags: Updated tags.

    Returns:
        JSON string with the updated space details.
    """
    try:
        backend = _get_backend()
        result = backend.update_space(
            space_id,
            name=name,
            description=description,
            allowed_agent_ids=allowed_agents,
            tags=tags,
        )
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def delete_space(space_id: str) -> str:
    """Delete a context space (owner only).

    Args:
        space_id: The space ID to delete.

    Returns:
        JSON string indicating success.
    """
    try:
        backend = _get_backend()
        backend.delete_space(space_id)
        return json.dumps({"success": True, "space_id": space_id})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def share_space(
    tag: str,
    email: str,
    permission: str = "read",
) -> str:
    """Share a space/tag with another user by email.

    Requires Novyx Cloud — not available in local mode.

    Args:
        tag: The tag or space tag to share.
        email: Email address of the recipient.
        permission: Access level — 'read' or 'write' (default 'read').

    Returns:
        JSON string with the share token and join URL.
    """
    try:
        backend = _get_backend()
        result = backend.share_context(tag, to_email=email, permission=permission)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sharing")


# =========================================================================
# Replay — Time-Travel Debugging (Pro+)
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def replay_timeline(
    since: str | None = None,
    until: str | None = None,
    operations: str | None = None,
    limit: int = 100,
) -> str:
    """Get the full timeline of memory operations. The tape you scrub through.

    Shows every create, update, delete, and rollback event with timestamps.
    Requires Pro tier or Novyx Cloud.

    Args:
        since: Start of time range (ISO timestamp).
        until: End of time range (ISO timestamp).
        operations: Comma-separated filter: create, update, delete, rollback.
        limit: Maximum entries to return (default 100).

    Returns:
        JSON string with timeline entries.
    """
    try:
        backend = _get_backend()
        op_list = [o.strip() for o in operations.split(",")] if operations else None
        result = backend.replay_timeline(since=since, until=until, operations=op_list, limit=limit)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Replay")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def replay_snapshot(
    at: str,
    limit: int = 500,
) -> str:
    """Reconstruct memory state at a specific point in time.

    Returns all memories and their link graph as they existed at timestamp T.
    Requires Pro tier or Novyx Cloud.

    Args:
        at: ISO timestamp to reconstruct state at.
        limit: Maximum memories to return (default 500).

    Returns:
        JSON string with memory snapshot and edges.
    """
    try:
        backend = _get_backend()
        result = backend.replay_snapshot(at, limit=limit)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Replay")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def replay_lifecycle(memory_id: str) -> str:
    """Full biography of a single memory.

    Shows creation, every update, every recall, every link, and deletion.
    Use this to understand why a memory exists and how it evolved.
    Requires Pro tier or Novyx Cloud.

    Args:
        memory_id: UUID of the memory to inspect.

    Returns:
        JSON string with the memory's full lifecycle events.
    """
    try:
        backend = _get_backend()
        result = backend.replay_lifecycle(memory_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Replay")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def replay_diff(
    start: str,
    end: str,
) -> str:
    """Diff memory state between two timestamps.

    Shows what was added, removed, and modified in a time range.
    Requires Pro tier or Novyx Cloud.

    Args:
        start: Start timestamp (ISO).
        end: End timestamp (ISO).

    Returns:
        JSON string with added, removed, and modified memories.
    """
    try:
        backend = _get_backend()
        result = backend.replay_diff(start, end)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Replay")


# =========================================================================
# Cortex — Autonomous Memory Intelligence (Pro+)
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def cortex_status() -> str:
    """Get Cortex autonomous intelligence status.

    Shows whether Cortex is enabled, last run time, and consolidation/reinforcement stats.
    Requires Pro tier or Novyx Cloud.

    Returns:
        JSON string with Cortex status and last run info.
    """
    try:
        backend = _get_backend()
        result = backend.cortex_status()
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Cortex")


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def cortex_run() -> str:
    """Manually trigger a Cortex cycle.

    Runs consolidation (merge duplicate memories) and reinforcement (boost frequently
    recalled memories, decay forgotten ones). Normally runs automatically every 6 hours.
    Requires Pro tier or Novyx Cloud.

    Returns:
        JSON string with cycle results (consolidated, reinforced counts).
    """
    try:
        backend = _get_backend()
        result = backend.cortex_run()
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Cortex")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def cortex_insights(limit: int = 20) -> str:
    """List auto-generated memory insights.

    Cortex detects patterns across your memories and generates insights automatically.
    Requires Enterprise tier or Novyx Cloud.

    Args:
        limit: Maximum insights to return (default 20).

    Returns:
        JSON string with generated insights.
    """
    try:
        backend = _get_backend()
        result = backend.cortex_insights(limit=limit)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Cortex Insights")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def cortex_config() -> str:
    """Get the current Cortex configuration.

    Shows consolidation thresholds, reinforcement decay rates, and cycle schedule.
    Requires Pro tier or Novyx Cloud.

    Returns:
        JSON string with Cortex configuration.
    """
    try:
        backend = _get_backend()
        result = backend.cortex_config()
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Cortex Config")


# =========================================================================
# Supersede
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def supersede(old_memory_id: str, new_memory_id: str) -> str:
    """Mark a memory as superseded by a newer one.

    The old memory remains in the system for audit purposes but is flagged
    as superseded. Use when information has been updated or corrected.

    Args:
        old_memory_id: UUID of the memory being replaced.
        new_memory_id: UUID of the replacement memory.

    Returns:
        JSON string confirming the supersede operation.
    """
    try:
        backend = _get_backend()
        result = backend.supersede(old_memory_id, new_memory_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =========================================================================
# Links (extended)
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def unlink(source_id: str, target_id: str) -> str:
    """Remove a link between two memories.

    Args:
        source_id: UUID of the source memory.
        target_id: UUID of the target memory.

    Returns:
        JSON string confirming link removal.
    """
    try:
        backend = _get_backend()
        result = backend.unlink(source_id, target_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_links(memory_id: str, relation: str | None = None) -> str:
    """Get all links for a memory.

    Shows both incoming and outgoing connections in the memory graph.

    Args:
        memory_id: UUID of the memory.
        relation: Optional relation type filter (e.g. "related", "causes", "supports").

    Returns:
        JSON string with links list.
    """
    try:
        backend = _get_backend()
        result = backend.links(memory_id, relation=relation)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def graph_edges(
    memory_id: str | None = None,
    relation: str | None = None,
    direction: str = "both",
    limit: int = 100,
) -> str:
    """Query the memory graph edges with filters.

    Browse the relationship graph between memories. Use to understand
    how memories are connected.

    Args:
        memory_id: Optional UUID to filter edges involving a specific memory.
        relation: Optional relation type filter.
        direction: "outgoing", "incoming", or "both" (default).
        limit: Maximum edges to return (default 100).

    Returns:
        JSON string with matching edges.
    """
    try:
        backend = _get_backend()
        result = backend.edges(
            memory_id=memory_id, relation=relation, direction=direction, limit=limit
        )
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =========================================================================
# Knowledge Graph (extended)
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def delete_triple(triple_id: str) -> str:
    """Delete a knowledge graph triple.

    Removes a subject-predicate-object relationship from the knowledge graph.

    Args:
        triple_id: ID of the triple to delete.

    Returns:
        JSON string confirming deletion.
    """
    try:
        backend = _get_backend()
        result = backend.delete_triple(triple_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_entities(
    limit: int = 100,
    offset: int = 0,
    entity_type: str | None = None,
) -> str:
    """List knowledge graph entities.

    Entities are the nodes in your knowledge graph — subjects and objects of triples.

    Args:
        limit: Maximum entities to return (default 100).
        offset: Pagination offset.
        entity_type: Optional type filter.

    Returns:
        JSON string with entities list.
    """
    try:
        backend = _get_backend()
        result = backend.entities(limit=limit, offset=offset, entity_type=entity_type)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_entity(entity_id: str) -> str:
    """Get a knowledge graph entity and its associated triples.

    Args:
        entity_id: ID of the entity.

    Returns:
        JSON string with entity details and related triples.
    """
    try:
        backend = _get_backend()
        result = backend.entity(entity_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def delete_entity(entity_id: str) -> str:
    """Delete a knowledge graph entity and all its triples.

    Removes the entity node and every triple where it appears as subject or object.

    Args:
        entity_id: ID of the entity to delete.

    Returns:
        JSON string confirming deletion.
    """
    try:
        backend = _get_backend()
        result = backend.delete_entity(entity_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =========================================================================
# Rollback (extended)
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def rollback_preview(target: str) -> str:
    """Preview what a rollback would do without executing it.

    Shows which operations would be undone if you rolled back to the target
    timestamp. Always use this before an actual rollback.

    Args:
        target: ISO timestamp or relative expression (e.g. "2 hours ago").

    Returns:
        JSON string with preview of operations to undo.
    """
    try:
        backend = _get_backend()
        result = backend.rollback_preview(target)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def rollback_history(limit: int = 50) -> str:
    """List past rollback operations.

    Shows when rollbacks were performed, what they targeted, and how many
    operations were undone.

    Args:
        limit: Maximum rollback events to return (default 50).

    Returns:
        JSON string with rollback history.
    """
    try:
        backend = _get_backend()
        result = backend.rollback_history(limit=limit)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =========================================================================
# Audit (extended)
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def audit_verify() -> str:
    """Verify the integrity of the audit trail.

    Checks the cryptographic hash chain (cloud) or entry consistency (local)
    to confirm no audit entries have been tampered with.

    Returns:
        JSON string with verification result.
    """
    try:
        backend = _get_backend()
        result = backend.audit_verify()
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =========================================================================
# Execution Traces
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def trace_create(name: str, metadata: str | None = None) -> str:
    """Create an execution trace to track a multi-step agent workflow.

    Start a trace before a complex operation, add steps as you go,
    then complete it. Traces provide a full audit of agent reasoning.

    Args:
        name: Name describing this trace (e.g. "research-and-summarize").
        metadata: Optional JSON string with additional metadata.

    Returns:
        JSON string with trace_id and status.
    """
    try:
        backend = _get_backend()
        kwargs = {}
        if metadata:
            kwargs["metadata"] = json.loads(metadata)
        result = backend.trace_create(name, **kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def trace_step(
    trace_id: str,
    step_name: str,
    input_data: str | None = None,
    output_data: str | None = None,
) -> str:
    """Add a step to an execution trace.

    Record each significant action during a traced workflow. Include
    input and output data to make the trace useful for debugging.

    Args:
        trace_id: ID of the active trace.
        step_name: Name of this step (e.g. "search-memories", "call-api").
        input_data: Optional JSON string with step input.
        output_data: Optional JSON string with step output.

    Returns:
        JSON string with step_id and status.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {}
        if input_data:
            kwargs["input_data"] = json.loads(input_data)
        if output_data:
            kwargs["output_data"] = json.loads(output_data)
        result = backend.trace_step(trace_id, step_name, **kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def trace_complete(trace_id: str) -> str:
    """Mark an execution trace as complete.

    Finalizes the trace. After completion, the trace and all its steps
    are immutable and available for audit.

    Args:
        trace_id: ID of the trace to complete.

    Returns:
        JSON string with completion status and step count.
    """
    try:
        backend = _get_backend()
        result = backend.trace_complete(trace_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def trace_verify(trace_id: str) -> str:
    """Verify an execution trace's integrity.

    Confirms all steps are present and the trace hasn't been tampered with.

    Args:
        trace_id: ID of the trace to verify.

    Returns:
        JSON string with verification result.
    """
    try:
        backend = _get_backend()
        result = backend.trace_verify(trace_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =========================================================================
# Memory Eval
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def eval_run(min_score: float | None = None) -> str:
    """Run a memory health evaluation.

    Scores your memory quality on a 0-1 scale based on staleness, conflicts,
    and superseded memories. Optionally acts as a CI gate with min_score.

    Args:
        min_score: Optional minimum score threshold. If set, returns pass/fail gate result.

    Returns:
        JSON string with score, breakdown, and optional gate result.
    """
    try:
        backend = _get_backend()
        result = backend.eval_run(min_score=min_score)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def eval_gate(min_score: float) -> str:
    """CI gate — pass or fail based on memory health score.

    Use in CI/CD pipelines to block deploys when memory quality degrades.

    Args:
        min_score: Minimum acceptable score (0.0 to 1.0).

    Returns:
        JSON string with gate result (passed/failed) and score.
    """
    try:
        backend = _get_backend()
        result = backend.eval_gate(min_score)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def eval_history(limit: int = 50) -> str:
    """List past memory evaluation runs.

    Track how memory quality has changed over time.

    Args:
        limit: Maximum evaluations to return (default 50).

    Returns:
        JSON string with evaluation history.
    """
    try:
        backend = _get_backend()
        result = backend.eval_history(limit=limit)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def eval_drift(days: int = 7) -> str:
    """Detect memory drift over a time period.

    Shows how many memories were created, deleted, and updated during
    the specified period. Useful for monitoring memory churn.

    Args:
        days: Number of days to look back (default 7).

    Returns:
        JSON string with drift metrics.
    """
    try:
        backend = _get_backend()
        result = backend.eval_drift(days=days)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =========================================================================
# Replay (extended)
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def replay_memory(memory_id: str) -> str:
    """Get the full history of a single memory.

    Shows every operation that affected this memory — creation, updates,
    links, supersedes — in chronological order.

    Args:
        memory_id: UUID of the memory.

    Returns:
        JSON string with chronological event list.
    """
    try:
        backend = _get_backend()
        result = backend.replay_memory(memory_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def replay_recall(query: str, at: str, limit: int = 5) -> str:
    """Time-travel recall — what would search have returned at a past timestamp?

    Reconstructs the memory state at a historical point and runs a semantic
    search against it. Powerful for understanding how agent context evolved.
    Requires Pro tier or Novyx Cloud.

    Args:
        query: Search query.
        at: ISO timestamp to search at.
        limit: Maximum results (default 5).

    Returns:
        JSON string with historical search results.
    """
    try:
        backend = _get_backend()
        result = backend.replay_recall(query, at, limit=limit)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Time-travel recall")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def replay_memory_drift(from_ts: str, to_ts: str) -> str:
    """Detect memory drift between two timestamps.

    Compares memory state at two points in time and shows what changed.
    Requires Pro tier or Novyx Cloud.

    Args:
        from_ts: Start ISO timestamp.
        to_ts: End ISO timestamp.

    Returns:
        JSON string with drift analysis.
    """
    try:
        backend = _get_backend()
        result = backend.replay_drift(from_ts, to_ts)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Replay drift")


# =========================================================================
# Context & Dashboard
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def context_now() -> str:
    """Get a snapshot of your current memory context.

    Returns recent memories, stats, and audit activity — a quick overview
    of what your agent knows right now.

    Returns:
        JSON string with context snapshot.
    """
    try:
        backend = _get_backend()
        result = backend.context_now()
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def dashboard() -> str:
    """Get a full dashboard overview.

    Combines stats, spaces, and recent activity into a single response.
    Use for periodic status checks or reporting.

    Returns:
        JSON string with dashboard data.
    """
    try:
        backend = _get_backend()
        result = backend.dashboard()
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =========================================================================
# Sharing (extended)
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def accept_shared_context(token: str) -> str:
    """Accept a shared context invitation.

    When another user shares a context space with you, use this to accept
    and gain access to their memories. Requires Novyx Cloud.

    Args:
        token: The sharing token from the invitation.

    Returns:
        JSON string with acceptance result.
    """
    try:
        backend = _get_backend()
        result = backend.accept_shared_context(token)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Shared contexts")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def shared_contexts() -> str:
    """List all shared contexts you have access to.

    Shows contexts shared with you and contexts you've shared with others.
    Requires Novyx Cloud.

    Returns:
        JSON string with shared contexts list.
    """
    try:
        backend = _get_backend()
        result = backend.shared_contexts()
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Shared contexts")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def revoke_shared_context(token: str) -> str:
    """Revoke a shared context invitation.

    Removes access for the recipient. Requires Novyx Cloud.

    Args:
        token: The sharing token to revoke.

    Returns:
        JSON string confirming revocation.
    """
    try:
        backend = _get_backend()
        result = backend.revoke_shared_context(token)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Shared contexts")


# =========================================================================
# Control Tools (requires NOVYX_CONTROL_URL + NOVYX_CONTROL_API_KEY)
# =========================================================================


def _control_request(method: str, path: str, body: dict | None = None) -> dict:
    """Make an HTTP request to Novyx Control."""
    import urllib.error
    import urllib.request

    control_url = os.environ.get("NOVYX_CONTROL_URL")
    control_key = os.environ.get("NOVYX_CONTROL_API_KEY")
    if not control_url or not control_key:
        return {"error": "Control not configured. Set NOVYX_CONTROL_URL and NOVYX_CONTROL_API_KEY."}

    url = f"{control_url.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {control_key}",
        "Content-Type": "application/json",
    }

    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode())
        except Exception:
            detail = str(e)
        return {"error": detail, "status": e.code}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_pending(limit: int = 20) -> str:
    """List pending Control approval requests.

    Shows actions submitted by agents that require human approval before execution.
    Requires Novyx Control to be configured (NOVYX_CONTROL_URL + NOVYX_CONTROL_API_KEY).

    Args:
        limit: Maximum approvals to return (default 20).

    Returns:
        JSON string with pending approvals list.
    """
    result = _control_request("GET", f"/v1/approvals?limit={limit}")
    return json.dumps(result, default=str)


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def approve_action(approval_id: str, approver_id: str, reason: str = "") -> str:
    """Approve a pending agent action in Novyx Control.

    Approves the action and triggers its execution against the target connector
    (GitHub, Slack, Linear, PagerDuty, or HTTP).

    Args:
        approval_id: The approval ID (e.g. apr_act_xxx).
        approver_id: Your operator ID (e.g. usr_operator_default).
        reason: Optional reason for approval.

    Returns:
        JSON string with the executed action details.
    """
    body = {"approver_id": approver_id, "decision": "approved", "reason": reason}
    result = _control_request("POST", f"/v1/approvals/{approval_id}/decision", body)
    return json.dumps(result, default=str)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def check_policy(connector: str = "", environment: str = "production") -> str:
    """Check the current Control policy profile.

    Shows which connectors require approval, risk tier rules, and auto-approve settings.

    Args:
        connector: Optional connector to check (e.g. github, slack).
        environment: Environment to check against (default: production).

    Returns:
        JSON string with policy profile and whether the connector requires approval.
    """
    result = _control_request("GET", "/v1/control/policies")
    if "error" not in result:
        profile = result.get("policy_profile", {})
        if connector:
            requires = connector in (profile.get("require_approval_connectors") or [])
            env_requires = environment in (profile.get("require_approval_environments") or [])
            result["connector_check"] = {
                "connector": connector,
                "environment": environment,
                "requires_approval": requires and env_requires,
            }
    return json.dumps(result, default=str)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_policies(enabled_only: bool = True) -> str:
    """List all active Control policies (built-in + custom).

    Shows which policies are enforced on action submissions, including
    tenant-defined custom policies created via create_policy.

    Args:
        enabled_only: If True (default), only show enabled policies.

    Returns:
        JSON string with list of policies, their source (builtin/custom), and status.
    """
    result = _control_request("GET", "/v1/control/policies")
    if "error" not in result and enabled_only:
        policies = result.get("policies", [])
        result["policies"] = [p for p in policies if p.get("enabled", True)]
    return json.dumps(result, default=str)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def create_policy(
    name: str,
    description: str = "",
    rules: list = [],
    step_types: list = ["ACTION"],
    whitelisted_domains: list = [],
) -> str:
    """Create a custom governance policy for your agent's actions.

    Define rules with regex patterns and severity levels. Policies are evaluated
    alongside built-in policies on every action submission. Requires Starter plan.

    Args:
        name: Policy name (alphanumeric, underscores, hyphens). E.g. "pii_protection".
        description: Human-readable description of what this policy enforces.
        rules: List of rule dicts. Each rule needs:
            - match: regex pattern to detect (e.g. "(ssn|social.security)")
            - severity: "critical", "high", "medium", or "low"
            - reason: (optional) violation message template, use {match} for matched text
            - context_requires: (optional) additional regex that must also match
            - confidence: (optional) 0.0-1.0, default 0.85
        step_types: Which step types to evaluate (default: ["ACTION"]).
        whitelisted_domains: Domains to skip evaluation for.

    Returns:
        JSON string confirming policy creation with version number.
    """
    config = {
        "name": name,
        "description": description,
        "rules": rules,
        "step_types": step_types,
        "whitelisted_domains": whitelisted_domains,
    }
    result = _control_request("POST", "/v1/control/policies", config)
    return json.dumps(result, default=str)


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def delete_policy(policy_name: str) -> str:
    """Disable a custom governance policy.

    Soft-deletes the policy so it no longer evaluates on action submissions.
    Built-in policies (FinancialSafety, DataExfiltration) cannot be deleted.

    Args:
        policy_name: Name of the custom policy to disable.

    Returns:
        JSON string confirming policy was disabled.
    """
    result = _control_request("DELETE", f"/v1/control/policies/{policy_name}")
    return json.dumps(result, default=str)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def memory_health() -> str:
    """Check the health of your agent's memory.

    Returns a health score (0-100), stale memory count, conflict count,
    and contradiction count. Use this to monitor memory quality over time.

    Returns:
        JSON string with health score and breakdown.
    """
    backend = _get_backend()

    try:
        stats = backend.stats()
    except Exception:
        return json.dumps({"score": 100, "total": 0, "message": "No memories yet"})

    total = (
        stats.get("total_memories", 0)
        if isinstance(stats, dict)
        else getattr(stats, "total_memories", 0)
    )
    if total == 0:
        return json.dumps({"score": 100, "total": 0, "message": "No memories yet"})

    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    stale = _count_stale_memories(backend, cutoff)

    conflicts = 0
    try:
        if hasattr(backend, "_conn"):
            row = backend._conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE deleted = 0 AND confidence < 0.5"
            ).fetchone()
            conflicts = row["c"] if row else 0
        else:
            memories = backend.memories(limit=500)
            conflicts = sum(1 for m in memories if (m.get("confidence") or 1.0) < 0.5)
    except Exception:
        pass

    contradictions = (
        stats.get("contradictions", 0)
        if isinstance(stats, dict)
        else getattr(stats, "contradictions", 0)
    )

    score = _compute_health_score(total, stale, conflicts)

    return json.dumps(
        {
            "score": score,
            "total": total,
            "stale": stale,
            "conflicts": conflicts,
            "contradictions": contradictions,
            "breakdown": {
                "stale_penalty": min(30, int((stale / total) * 100)) if total else 0,
                "conflict_penalty": min(20, conflicts * 5),
            },
        }
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def action_history(limit: int = 20) -> str:
    """List recent Control actions with their status.

    Shows submitted, pending, approved, denied, executed, and failed actions.

    Args:
        limit: Maximum actions to return (default 20).

    Returns:
        JSON string with action list.
    """
    result = _control_request("GET", f"/v1/actions?limit={limit}")
    return json.dumps(result, default=str)


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def action_submit(
    connector: str,
    operation: str,
    payload: str,
) -> str:
    """Submit an action to Novyx Control for governed execution.

    The action is evaluated against your tenant's policy profile. If approval
    is required, it will be queued for human review; otherwise executed immediately.

    Args:
        connector: Target connector (github, slack, linear, pagerduty, http).
        operation: Operation name (e.g. issues/create, messages/send).
        payload: JSON string with the full action envelope.

    Returns:
        JSON string with action ID, status, and policy decision.
    """
    try:
        body = json.loads(payload)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON payload: {e}"})
    result = _control_request("POST", f"/v1/actions/{connector}/{operation}", body)
    return json.dumps(result, default=str)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def action_status(action_id: str) -> str:
    """Get the status of a specific Control action.

    Returns full action details including policy decision, approval status,
    execution result, and evidence (Novyx trace, certificate).

    Args:
        action_id: The action ID (e.g. act_xxx).

    Returns:
        JSON string with action details.
    """
    result = _control_request("GET", f"/v1/actions/{action_id}")
    return json.dumps(result, default=str)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def explain_action(action_id: str) -> str:
    """Get the full causal chain for a Control action.

    Returns what the agent recalled, what policy fired, what memory state
    existed at that moment, and the audit trail — all in one call.
    Requires Novyx Cloud.

    Args:
        action_id: The action UUID to explain.

    Returns:
        JSON string with explanation (memories, policy, audit trail).
    """
    try:
        backend = _get_backend()
        result = backend.explain_action(action_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Action explanations")


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def eval_baseline_create(query: str, expected_observation: str) -> str:
    """Save a recall baseline for regression testing.

    Future eval runs will check if this query still returns the expected result.
    Free: 1 baseline, Starter: 5, Pro: unlimited.

    Args:
        query: The recall query to baseline.
        expected_observation: Expected top result observation.

    Returns:
        JSON string with the created baseline.
    """
    try:
        backend = _get_backend()
        result = backend.eval_baseline_create(query, expected_observation)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Eval baselines")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def eval_baselines() -> str:
    """List all saved eval baselines.

    Shows the recall queries and expected observations used for regression testing.

    Returns:
        JSON string with baselines list.
    """
    try:
        backend = _get_backend()
        result = backend.eval_baselines()
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Eval baselines")


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def eval_baseline_delete(baseline_id: str) -> str:
    """Delete an eval baseline.

    Args:
        baseline_id: The baseline UUID to delete.

    Returns:
        JSON string confirming deletion.
    """
    try:
        backend = _get_backend()
        result = backend.eval_baseline_delete(baseline_id)
        return json.dumps({"deleted": result}, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Eval baselines")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def audit_export(format: str = "json") -> str:
    """Export the full audit log.

    Downloads the complete audit trail in the specified format.
    Requires Pro tier or Novyx Cloud.

    Args:
        format: Export format — "json", "csv", or "jsonl" (default: json).

    Returns:
        JSON string with exported audit data.
    """
    try:
        backend = _get_backend()
        result = backend.audit_export(format=format)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Audit export")


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def cortex_update_config(
    consolidation_threshold: float | None = None,
    reinforcement_boost: float | None = None,
    decay_rate: float | None = None,
) -> str:
    """Update Cortex configuration.

    Tune how Cortex consolidates, reinforces, and decays memories.
    Requires Pro tier or Novyx Cloud.

    Args:
        consolidation_threshold: Similarity threshold for merging duplicates (0.0-1.0).
        reinforcement_boost: How much to boost frequently recalled memories.
        decay_rate: How fast forgotten memories fade.

    Returns:
        JSON string with updated Cortex configuration.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {}
        if consolidation_threshold is not None:
            kwargs["consolidation_threshold"] = consolidation_threshold
        if reinforcement_boost is not None:
            kwargs["reinforcement_boost"] = reinforcement_boost
        if decay_rate is not None:
            kwargs["decay_rate"] = decay_rate
        result = backend.cortex_update_config(**kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Cortex Config")


# =========================================================================
# Sentinel Intel — Threat Intelligence (Pro+)
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def threat_feed(hours: int = 24, min_severity: str = "medium") -> str:
    """Get the anonymized threat intelligence feed.

    Shows attack patterns detected across the Novyx network, anonymized
    to protect individual tenants. Requires Pro tier or Novyx Cloud.

    Args:
        hours: Look back window in hours (default 24).
        min_severity: Minimum severity to include — low, medium, high, critical (default medium).

    Returns:
        JSON string with threat signatures.
    """
    try:
        backend = _get_backend()
        result = backend.threat_feed(hours=hours, min_severity=min_severity)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def threat_stats() -> str:
    """Get overall threat intelligence statistics.

    Shows total signatures, active threats, mitigated count, and severity breakdown.
    Requires Pro tier or Novyx Cloud.

    Returns:
        JSON string with threat stats.
    """
    try:
        backend = _get_backend()
        result = backend.threat_stats()
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def threat_record(threat_event: str) -> str:
    """Record a threat event for cross-tenant intelligence.

    The event is fingerprinted, deduplicated, and added to the threat network.
    Requires Pro tier or Novyx Cloud.

    Args:
        threat_event: JSON string describing the threat (pattern_type, details, severity).

    Returns:
        JSON string with the created/updated threat signature.
    """
    try:
        event = json.loads(threat_event)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    try:
        backend = _get_backend()
        result = backend.threat_record(event)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def threat_trending(hours: int = 24, min_occurrences: int = 2) -> str:
    """Get trending threat signatures.

    Shows the most active threats in the specified time window.
    Requires Pro tier or Novyx Cloud.

    Args:
        hours: Look back window in hours (default 24).
        min_occurrences: Minimum occurrence count to include (default 2).

    Returns:
        JSON string with trending threat signatures.
    """
    try:
        backend = _get_backend()
        result = backend.threat_trending(hours=hours, min_occurrences=min_occurrences)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def threat_match(threat_event: str, min_similarity: float = 0.8) -> str:
    """Find known threat signatures matching a threat event.

    Compares the event fingerprint against the threat database.
    Requires Pro tier or Novyx Cloud.

    Args:
        threat_event: JSON string describing the threat to match.
        min_similarity: Minimum similarity threshold 0.0-1.0 (default 0.8).

    Returns:
        JSON string with matching signatures.
    """
    try:
        event = json.loads(threat_event)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    try:
        backend = _get_backend()
        result = backend.threat_match(event, min_similarity=min_similarity)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def threat_signature(signature_id: str) -> str:
    """Get a specific threat signature by ID.

    Returns full details including pattern hash, severity, occurrence count,
    and whether a defense has been deployed. Requires Pro tier or Novyx Cloud.

    Args:
        signature_id: The threat signature ID.

    Returns:
        JSON string with signature details.
    """
    try:
        backend = _get_backend()
        result = backend.threat_signature(signature_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def threat_mitigate(signature_id: str) -> str:
    """Mark a threat signature as mitigated.

    Indicates that the threat has been addressed. Requires Pro tier or Novyx Cloud.

    Args:
        signature_id: The threat signature ID to mark as mitigated.

    Returns:
        JSON string confirming mitigation.
    """
    try:
        backend = _get_backend()
        result = backend.threat_mitigate(signature_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


# =========================================================================
# Sentinel Intel — Auto Defense (Pro+)
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def defense_list(rule_type: str | None = None) -> str:
    """List active auto-deployed defense rules.

    Shows all defense rules currently protecting against detected threats.
    Requires Pro tier or Novyx Cloud.

    Args:
        rule_type: Optional filter — block, rate_limit, quarantine, or alert_only.

    Returns:
        JSON string with active defenses.
    """
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {}
        if rule_type is not None:
            kwargs["rule_type"] = rule_type
        result = backend.defense_list(**kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def defense_deploy(signature_id: str, rule_type: str, rule_config: str | None = None) -> str:
    """Deploy a defense rule against a threat signature.

    Automatically blocks, rate-limits, quarantines, or alerts on matching threats.
    Requires Pro tier or Novyx Cloud.

    Args:
        signature_id: The threat signature to defend against.
        rule_type: Defense type — block, rate_limit, quarantine, or alert_only.
        rule_config: Optional JSON string with rule configuration.

    Returns:
        JSON string with the deployed defense details.
    """
    config = None
    if rule_config:
        try:
            config = json.loads(rule_config)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid rule_config JSON: {e}"})
    try:
        backend = _get_backend()
        result = backend.defense_deploy(signature_id, rule_type, config)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def defense_remove(defense_id: str) -> str:
    """Remove a deployed defense rule.

    Stops the defense from blocking/rate-limiting threats.
    Requires Pro tier or Novyx Cloud.

    Args:
        defense_id: The defense rule ID to remove.

    Returns:
        JSON string confirming removal.
    """
    try:
        backend = _get_backend()
        result = backend.defense_remove(defense_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def defense_effectiveness(defense_id: str) -> str:
    """Measure the effectiveness of a deployed defense.

    Returns a 0-1 score based on blocks vs false positives.
    Requires Pro tier or Novyx Cloud.

    Args:
        defense_id: The defense rule ID to evaluate.

    Returns:
        JSON string with effectiveness score.
    """
    try:
        backend = _get_backend()
        result = backend.defense_effectiveness(defense_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def defense_record_block(defense_id: str, is_false_positive: bool = False) -> str:
    """Record that a defense blocked a threat.

    Used to track defense performance. Mark false positives to auto-tune rules.
    Requires Pro tier or Novyx Cloud.

    Args:
        defense_id: The defense rule that blocked the threat.
        is_false_positive: Whether this block was a false positive (default False).

    Returns:
        JSON string confirming the record.
    """
    try:
        backend = _get_backend()
        result = backend.defense_record_block(defense_id, is_false_positive=is_false_positive)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def defense_stats() -> str:
    """Get overall defense statistics.

    Shows total defenses, active count, average effectiveness, and block counts.
    Requires Pro tier or Novyx Cloud.

    Returns:
        JSON string with defense statistics.
    """
    try:
        backend = _get_backend()
        result = backend.defense_stats()
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def defense_recommend(signature_id: str) -> str:
    """Get a recommended defense strategy for a threat signature.

    Returns the suggested rule type, confidence level, and reasoning.
    Requires Pro tier or Novyx Cloud.

    Args:
        signature_id: The threat signature to get a recommendation for.

    Returns:
        JSON string with recommendation (rule_type, confidence, reasoning).
    """
    try:
        backend = _get_backend()
        result = backend.defense_recommend(signature_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


# =========================================================================
# Sentinel Intel — Correlation (Pro+)
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def correlate_threat(threat_event: str) -> str:
    """Check if a threat correlates with attacks on other tenants.

    Cross-references the threat fingerprint against the network database.
    Requires Pro tier or Novyx Cloud.

    Args:
        threat_event: JSON string describing the threat to correlate.

    Returns:
        JSON string with correlation result and matching signatures.
    """
    try:
        event = json.loads(threat_event)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    try:
        backend = _get_backend()
        result = backend.correlate_threat(event)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def detect_campaign(hours: int = 24) -> str:
    """Detect an ongoing attack campaign.

    Looks for coordinated attack patterns across multiple tenants.
    Returns campaign details if detected. Requires Pro tier or Novyx Cloud.

    Args:
        hours: Look back window in hours (default 24).

    Returns:
        JSON string with campaign details or campaign_detected: false.
    """
    try:
        backend = _get_backend()
        result = backend.detect_campaign(hours=hours)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def coordinated_attack_check(threat_events: str, time_window_hours: int | None = None) -> str:
    """Check if multiple threat events are part of a coordinated attack.

    Analyzes timing and pattern similarity across events.
    Requires Pro tier or Novyx Cloud.

    Args:
        threat_events: JSON string with array of threat events to analyze.
        time_window_hours: Optional time window for correlation (default: auto).

    Returns:
        JSON string with is_coordinated boolean.
    """
    try:
        events = json.loads(threat_events)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    try:
        backend = _get_backend()
        kwargs: dict[str, Any] = {}
        if time_window_hours is not None:
            kwargs["time_window_hours"] = time_window_hours
        result = backend.coordinated_attack_check(events, **kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def related_signatures(signature_id: str, similarity_threshold: float = 0.7) -> str:
    """Find threat signatures related to a given signature.

    Uses pattern similarity to find related threats.
    Requires Pro tier or Novyx Cloud.

    Args:
        signature_id: The signature ID to find related threats for.
        similarity_threshold: Minimum similarity 0.0-1.0 (default 0.7).

    Returns:
        JSON string with related signatures.
    """
    try:
        backend = _get_backend()
        result = backend.related_signatures(signature_id, similarity_threshold=similarity_threshold)
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Sentinel Intel")


# =========================================================================
# Streams
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def stream_status() -> str:
    """Get real-time memory stream connection status.

    Shows active connections, max allowed, and event bus metrics.
    Requires Novyx Cloud.

    Returns:
        JSON string with stream status.
    """
    try:
        backend = _get_backend()
        result = backend.stream_status()
        return json.dumps(result, default=str)
    except Exception as e:
        return _handle_tier_error(e, "Streams")


# =========================================================================
# Resources
# =========================================================================


@mcp.resource("novyx://memories")
def resource_memories() -> str:
    """List all stored memories."""
    try:
        backend = _get_backend()
        result = backend.memories(limit=100)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("novyx://memories/{memory_id}")
def resource_memory(memory_id: str) -> str:
    """Get a specific memory by its UUID."""
    try:
        backend = _get_backend()
        result = backend.memory(memory_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("novyx://stats")
def resource_stats() -> str:
    """Get memory statistics for the current account."""
    try:
        backend = _get_backend()
        result = backend.stats()
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("novyx://usage")
def resource_usage() -> str:
    """Get usage and plan information for the current account."""
    try:
        backend = _get_backend()
        result = backend.usage()
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("novyx://spaces")
def resource_spaces() -> str:
    """List all context spaces accessible to the current tenant."""
    try:
        backend = _get_backend()
        result = backend.list_spaces()
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("novyx://spaces/{space_id}")
def resource_space(space_id: str) -> str:
    """Get a specific context space with memory count."""
    try:
        backend = _get_backend()
        result = backend.get_space(space_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =========================================================================
# Prompts
# =========================================================================


@mcp.prompt()
def memory_context(query: str) -> str:
    """Recall relevant memories and format them as context for the current conversation.

    Args:
        query: The topic or question to find relevant memories for.
    """
    try:
        backend = _get_backend()
        result = backend.recall(query, limit=10)

        # Handle both dict (local) and SearchResult (cloud) responses
        if isinstance(result, dict):
            mems = result.get("memories", [])
            if not mems:
                return f"No relevant memories found for: {query}"

            lines = [f"## Relevant Memories for: {query}\n"]
            for i, mem in enumerate(mems, 1):
                score_str = f"{mem.get('score', 0):.2f}"
                tags_str = ", ".join(mem.get("tags", [])) or "none"
                lines.append(
                    f"{i}. **{mem['observation']}**\n"
                    f"   - Score: {score_str} | Importance: {mem.get('importance', 5)} | Tags: {tags_str}\n"
                    f"   - ID: {mem['uuid']} | Created: {mem.get('created_at', '')}\n"
                )
            return "\n".join(lines)

        # Cloud mode: SearchResult object
        if not result.memories:
            return f"No relevant memories found for: {query}"

        lines = [f"## Relevant Memories for: {query}\n"]
        for i, mem in enumerate(result.memories, 1):
            score_str = f"{mem.score:.2f}" if mem.score is not None else "N/A"
            tags_str = ", ".join(mem.tags) if mem.tags else "none"
            lines.append(
                f"{i}. **{mem.observation}**\n"
                f"   - Score: {score_str} | Importance: {mem.importance} | Tags: {tags_str}\n"
                f"   - ID: {mem.uuid} | Created: {mem.created_at}\n"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error recalling memories: {e}"


@mcp.prompt()
def session_summary(session_id: str) -> str:
    """List all memories tagged with a specific session ID.

    Args:
        session_id: The session identifier to look up.
    """
    try:
        backend = _get_backend()
        tag = f"session:{session_id}"
        result = backend.memories(tags=[tag], limit=100)

        if not result:
            return f"No memories found for session: {session_id}"

        lines = [f"## Session Summary: {session_id}\n"]
        lines.append(f"Total memories: {len(result)}\n")
        for i, mem in enumerate(result, 1):
            obs = mem.get("observation", "")
            tags = ", ".join(mem.get("tags", []))
            created = mem.get("created_at", "")
            lines.append(f"{i}. {obs}\n   - Tags: {tags} | Created: {created}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching session summary: {e}"


@mcp.prompt()
def space_context(space_id: str, query: str = "") -> str:
    """Recall memories from a shared context space and format them as context.

    Args:
        space_id: The context space to query.
        query: Optional topic to search for within the space.
    """
    try:
        backend = _get_backend()
        result = backend.space_memories(space_id, query=query or None, limit=20)
        memories = result.get("memories", [])

        if not memories:
            return f"No memories found in space {space_id}" + (f" for: {query}" if query else "")

        lines = [f"## Context Space: {space_id}\n"]
        if query:
            lines[0] += f" (query: {query})\n"
        lines.append(f"Total: {result.get('total_count', len(memories))} memories\n")
        for i, mem in enumerate(memories, 1):
            obs = mem.get("observation", "")
            tags_str = ", ".join(mem.get("tags", [])) if mem.get("tags") else "none"
            lines.append(
                f"{i}. **{obs}**\n   - Tags: {tags_str} | Created: {mem.get('created_at', '')}\n"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error querying space: {e}"


# =========================================================================
# Startup Health Check
# =========================================================================


def _count_stale_memories(backend: Any, cutoff_iso: str) -> int:
    """Count memories not updated since *cutoff_iso*.

    Works for both local (SQLite) and cloud backends.
    """
    # Local backend — direct SQL for speed
    if hasattr(backend, "_conn"):
        row = backend._conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE deleted = 0 "
            "AND COALESCE(updated_at, created_at) < ?",
            (cutoff_iso,),
        ).fetchone()
        return row["c"] if row else 0

    # Cloud backend — fetch recent memories and compare timestamps
    try:
        memories = backend.memories(limit=500)
        stale = 0
        for mem in memories:
            ts = mem.get("updated_at") or mem.get("created_at", "")
            if ts and ts < cutoff_iso:
                stale += 1
        return stale
    except Exception:
        return 0


def _compute_health_score(total: int, stale: int, conflicts: int) -> int:
    """Compute a 0-100 health score.

    Scoring:
    - Start at 100
    - Lose up to 30 points for stale ratio (stale / total)
    - Lose up to 20 points for conflicts
    - Minimum score is 0
    """
    if total == 0:
        return 100

    stale_ratio = stale / total
    stale_penalty = min(30, int(stale_ratio * 100))
    conflict_penalty = min(20, conflicts * 5)

    return max(0, 100 - stale_penalty - conflict_penalty)


def startup_health_check() -> str | None:
    """Run a quick health evaluation and return a one-line summary.

    Returns the summary string, or None if there are no memories yet.
    Also prints the summary to stderr.
    """
    try:
        backend = _get_backend()
        stats = backend.stats()
    except Exception:
        msg = "Novyx: ready (no memories yet)"
        print(msg, file=sys.stderr)
        return msg

    total = 0
    if isinstance(stats, dict):
        total = stats.get("total_memories", 0)
    else:
        # Fallback: treat stats as object with attribute
        total = getattr(stats, "total_memories", 0)

    if total == 0:
        msg = "Novyx: ready (no memories yet)"
        print(msg, file=sys.stderr)
        return msg

    # Stale = not updated in 30+ days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    stale = _count_stale_memories(backend, cutoff)

    # Conflicts: check for memories with low confidence as a proxy
    conflicts = 0
    try:
        if hasattr(backend, "_conn"):
            row = backend._conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE deleted = 0 AND confidence < 0.5"
            ).fetchone()
            conflicts = row["c"] if row else 0
        else:
            memories = backend.memories(limit=500)
            conflicts = sum(1 for m in memories if (m.get("confidence") or 1.0) < 0.5)
    except Exception:
        pass

    score = _compute_health_score(total, stale, conflicts)

    parts = []
    if stale:
        parts.append(f"{stale} stale")
    if conflicts:
        parts.append(f"{conflicts} conflicts")
    detail = f" ({', '.join(parts)})" if parts else ""

    msg = f"Novyx: {total} memories, health {score}/100{detail}"
    print(msg, file=sys.stderr)
    return msg


# =========================================================================
# Runtime v2: Agents
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def create_agent(
    name: str,
    provider: str,
    model: str,
    agent_id: str | None = None,
    description: str | None = None,
    instructions: str | None = None,
    capabilities: list[str] | None = None,
    memory_scope: str | None = None,
    policy_profile: str | None = None,
) -> str:
    """Create a first-class agent entity in the Novyx Runtime.

    Novyx is provider-agnostic — you must specify which LLM backend the
    agent uses. Use "litellm" to reach models not directly supported
    (Gemini, Mistral, Cohere, local Ollama, etc.).

    Args:
        name: Human-readable agent name.
        provider: LLM provider (required): "openai", "anthropic", or "litellm".
        model: LLM model name (required, e.g. "gpt-4o", "claude-sonnet-4-6").
        agent_id: Custom agent ID (auto-generated if omitted).
        description: Agent description.
        instructions: System prompt / instructions.
        capabilities: List of enabled capability pack names.
        memory_scope: Memory scope for the agent (e.g. "private", "shared").
        policy_profile: JSON string of policy profile configuration.
    """
    if provider not in ("openai", "anthropic", "litellm"):
        return json.dumps({
            "error": (
                f"Invalid provider '{provider}'. "
                "Choose 'openai', 'anthropic', or 'litellm' "
                "(use litellm for Gemini, Mistral, Cohere, etc.)."
            )
        })
    if not model:
        return json.dumps({"error": "model is required"})

    kwargs: dict[str, Any] = {"name": name, "model": model, "provider": provider}
    if agent_id:
        kwargs["agent_id"] = agent_id
    if description:
        kwargs["description"] = description
    if instructions:
        kwargs["instructions"] = instructions
    if capabilities:
        kwargs["capabilities"] = capabilities
    if memory_scope:
        kwargs["memory_scope"] = memory_scope
    if policy_profile:
        try:
            kwargs["policy_profile"] = json.loads(policy_profile)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid policy_profile JSON: {e}"})
    return _call_backend_json("create_agent", **kwargs)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_agents(status: str | None = None, limit: int = 100) -> str:
    """List all agents for the current tenant.

    Args:
        status: Filter by status (active, paused, archived).
        limit: Max results.
    """
    kwargs: dict[str, Any] = {"limit": limit}
    if status:
        kwargs["status"] = status
    return _call_backend_json("list_agents", **kwargs)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_agent(agent_id: str) -> str:
    """Get an agent by ID."""
    return _call_backend_json("get_agent", agent_id)


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def delete_agent(agent_id: str) -> str:
    """Delete an agent."""
    return _call_backend_json("delete_agent", agent_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
def update_agent(
    agent_id: str,
    name: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    instructions: str | None = None,
    capabilities: list[str] | None = None,
    memory_scope: str | None = None,
    policy_profile: str | None = None,
    config: str | None = None,
) -> str:
    """Update an existing agent's configuration.

    Args:
        agent_id: Agent to update.
        name: New agent name.
        model: New LLM model name.
        provider: New LLM provider.
        instructions: New system prompt / instructions.
        capabilities: New list of enabled capability pack names.
        memory_scope: New memory scope.
        policy_profile: JSON string of policy profile configuration.
        config: JSON string of additional configuration.
    """
    kwargs: dict[str, Any] = {}
    if name:
        kwargs["name"] = name
    if model:
        kwargs["model"] = model
    if provider:
        kwargs["provider"] = provider
    if instructions:
        kwargs["instructions"] = instructions
    if capabilities:
        kwargs["capabilities"] = capabilities
    if memory_scope:
        kwargs["memory_scope"] = memory_scope
    if policy_profile:
        try:
            kwargs["policy_profile"] = json.loads(policy_profile)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid policy_profile JSON: {e}"})
    if config:
        try:
            kwargs["config"] = json.loads(config)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid config JSON: {e}"})
    return _call_backend_json("update_agent", agent_id, **kwargs)


# =========================================================================
# Runtime v2: Missions
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def create_mission(
    agent_id: str,
    goal: str,
    constraints: list[str] | None = None,
    success_criteria: list[str] | None = None,
    allowed_capabilities: list[str] | None = None,
) -> str:
    """Create a mission (bounded job) for an agent.

    Args:
        agent_id: Agent to assign this mission to.
        goal: What the mission should accomplish.
        constraints: Constraints on execution.
        success_criteria: How to determine success.
        allowed_capabilities: Capability packs allowed.
    """
    kwargs: dict[str, Any] = {"agent_id": agent_id, "goal": goal}
    if constraints:
        kwargs["constraints"] = constraints
    if success_criteria:
        kwargs["success_criteria"] = success_criteria
    if allowed_capabilities:
        kwargs["allowed_capabilities"] = allowed_capabilities
    return _call_backend_json("create_mission", **kwargs)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_missions(agent_id: str | None = None, status: str | None = None, limit: int = 100) -> str:
    """List missions for the current tenant.

    Args:
        agent_id: Filter by agent.
        status: Filter by status (queued, running, paused, completed, failed).
        limit: Max results.
    """
    kwargs: dict[str, Any] = {"limit": limit}
    if agent_id:
        kwargs["agent_id"] = agent_id
    if status:
        kwargs["status"] = status
    return _call_backend_json("list_missions", **kwargs)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_mission(mission_id: str) -> str:
    """Get a mission by ID."""
    return _call_backend_json("get_mission", mission_id)


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def pause_mission(mission_id: str) -> str:
    """Pause a running mission."""
    return _call_backend_json("pause_mission", mission_id)


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def resume_mission(mission_id: str) -> str:
    """Resume a paused mission."""
    return _call_backend_json("resume_mission", mission_id)


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def cancel_mission(mission_id: str) -> str:
    """Cancel a mission."""
    return _call_backend_json("cancel_mission", mission_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
def update_mission(
    mission_id: str,
    goal: str | None = None,
    constraints: list[str] | None = None,
    success_criteria: list[str] | None = None,
    allowed_capabilities: list[str] | None = None,
    escalation_rules: str | None = None,
    stop_conditions: str | None = None,
    config: str | None = None,
) -> str:
    """Update an existing mission's configuration.

    Args:
        mission_id: Mission to update.
        goal: New mission goal.
        constraints: New constraints on execution.
        success_criteria: New success criteria.
        allowed_capabilities: New allowed capability packs.
        escalation_rules: JSON string of escalation rules.
        stop_conditions: JSON string of stop conditions.
        config: JSON string of additional configuration.
    """
    kwargs: dict[str, Any] = {}
    if goal:
        kwargs["goal"] = goal
    if constraints:
        kwargs["constraints"] = constraints
    if success_criteria:
        kwargs["success_criteria"] = success_criteria
    if allowed_capabilities:
        kwargs["allowed_capabilities"] = allowed_capabilities
    if escalation_rules:
        try:
            kwargs["escalation_rules"] = json.loads(escalation_rules)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid escalation_rules JSON: {e}"})
    if stop_conditions:
        try:
            kwargs["stop_conditions"] = json.loads(stop_conditions)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid stop_conditions JSON: {e}"})
    if config:
        try:
            kwargs["config"] = json.loads(config)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid config JSON: {e}"})
    return _call_backend_json("update_mission", mission_id, **kwargs)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def delete_mission(mission_id: str) -> str:
    """Delete a mission."""
    return _call_backend_json("delete_mission", mission_id)


# =========================================================================
# Runtime v2: Capabilities
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def create_capability(
    name: str,
    description: str | None = None,
    tools: list[dict] | None = None,
    risk_levels: dict | None = None,
    approval_requirements: str | None = None,
    memory_behavior: str | None = None,
    eval_rules: str | None = None,
) -> str:
    """Register a capability pack (tool bundle with governance).

    Args:
        name: Capability pack name.
        description: Description.
        tools: Tool definitions with schemas.
        risk_levels: Risk level per tool {tool_name: "low"|"medium"|"high"|"critical"}.
        approval_requirements: JSON string of approval requirements per risk level.
        memory_behavior: JSON string of memory behavior configuration.
        eval_rules: JSON string of evaluation rules.
    """
    kwargs: dict[str, Any] = {"name": name}
    if description:
        kwargs["description"] = description
    if tools:
        kwargs["tools"] = tools
    if risk_levels:
        kwargs["risk_levels"] = risk_levels
    if approval_requirements:
        try:
            kwargs["approval_requirements"] = json.loads(approval_requirements)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid approval_requirements JSON: {e}"})
    if memory_behavior:
        try:
            kwargs["memory_behavior"] = json.loads(memory_behavior)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid memory_behavior JSON: {e}"})
    if eval_rules:
        try:
            kwargs["eval_rules"] = json.loads(eval_rules)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid eval_rules JSON: {e}"})
    return _call_backend_json("create_capability", **kwargs)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_capabilities(status: str | None = None) -> str:
    """List registered capability packs."""
    kwargs: dict[str, Any] = {}
    if status:
        kwargs["status"] = status
    return _call_backend_json("list_capabilities", **kwargs)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def get_capability(capability_id: str) -> str:
    """Get a capability pack by ID."""
    return _call_backend_json("get_capability", capability_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
def update_capability(
    capability_id: str,
    name: str | None = None,
    description: str | None = None,
    tools: list[dict] | None = None,
    risk_levels: dict | None = None,
    approval_requirements: str | None = None,
    memory_behavior: str | None = None,
    eval_rules: str | None = None,
    config: str | None = None,
    status: str | None = None,
) -> str:
    """Update an existing capability pack.

    Args:
        capability_id: Capability to update.
        name: New capability name.
        description: New description.
        tools: New tool definitions.
        risk_levels: New risk levels per tool.
        approval_requirements: JSON string of approval requirements.
        memory_behavior: JSON string of memory behavior configuration.
        eval_rules: JSON string of evaluation rules.
        config: JSON string of additional configuration.
        status: New status (active, deprecated).
    """
    kwargs: dict[str, Any] = {}
    if name:
        kwargs["name"] = name
    if description:
        kwargs["description"] = description
    if tools:
        kwargs["tools"] = tools
    if risk_levels:
        kwargs["risk_levels"] = risk_levels
    if approval_requirements:
        try:
            kwargs["approval_requirements"] = json.loads(approval_requirements)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid approval_requirements JSON: {e}"})
    if memory_behavior:
        try:
            kwargs["memory_behavior"] = json.loads(memory_behavior)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid memory_behavior JSON: {e}"})
    if eval_rules:
        try:
            kwargs["eval_rules"] = json.loads(eval_rules)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid eval_rules JSON: {e}"})
    if config:
        try:
            kwargs["config"] = json.loads(config)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid config JSON: {e}"})
    if status:
        kwargs["status"] = status
    return _call_backend_json("update_capability", capability_id, **kwargs)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def delete_capability(capability_id: str) -> str:
    """Delete a capability pack."""
    return _call_backend_json("delete_capability", capability_id)


# =========================================================================
# Runtime v2: Checkpoints
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def create_checkpoint(
    mission_id: str,
    label: str | None = None,
    metadata: str | None = None,
) -> str:
    """Create a checkpoint for a mission (rollback point).

    Args:
        mission_id: Mission to checkpoint.
        label: Human-readable label.
        metadata: JSON string of additional metadata.
    """
    kwargs: dict[str, Any] = {"mission_id": mission_id}
    if label:
        kwargs["label"] = label
    if metadata:
        try:
            kwargs["metadata"] = json.loads(metadata)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid metadata JSON: {e}"})
    return _call_backend_json("create_checkpoint", **kwargs)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def get_checkpoint(checkpoint_id: str) -> str:
    """Get a checkpoint by ID."""
    return _call_backend_json("get_checkpoint", checkpoint_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_checkpoints(mission_id: str) -> str:
    """List checkpoints for a mission."""
    return _call_backend_json("list_checkpoints", mission_id)


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def rollback_to_checkpoint(mission_id: str, checkpoint_id: str, reason: str | None = None) -> str:
    """Rollback a mission to a previous checkpoint.

    Args:
        mission_id: Mission to rollback.
        checkpoint_id: Target checkpoint.
        reason: Why the rollback is needed.
    """
    kwargs: dict[str, Any] = {"mission_id": mission_id, "checkpoint_id": checkpoint_id}
    if reason:
        kwargs["reason"] = reason
    return _call_backend_json("rollback_to_checkpoint", **kwargs)


# =========================================================================
# Runtime v2: Interventions
# =========================================================================


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
def create_intervention(
    intervention_type: str,
    mission_id: str | None = None,
    action_id: str | None = None,
    agent_id: str | None = None,
    rationale: str | None = None,
    metadata: str | None = None,
) -> str:
    """Record a supervisor intervention (approve, reject, pause, escalate, etc).

    Args:
        intervention_type: One of: approve, reject, pause, escalate, reroute, annotate, rollback_request.
        mission_id: Related mission.
        action_id: Related action.
        agent_id: Related agent.
        rationale: Why this intervention was made.
        metadata: JSON string of additional metadata.
    """
    kwargs: dict[str, Any] = {"intervention_type": intervention_type}
    if mission_id:
        kwargs["mission_id"] = mission_id
    if action_id:
        kwargs["action_id"] = action_id
    if agent_id:
        kwargs["agent_id"] = agent_id
    if rationale:
        kwargs["rationale"] = rationale
    if metadata:
        try:
            kwargs["metadata"] = json.loads(metadata)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid metadata JSON: {e}"})
    return _call_backend_json("create_intervention", **kwargs)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def get_intervention(intervention_id: str) -> str:
    """Get a supervisor intervention by ID."""
    return _call_backend_json("get_intervention", intervention_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_interventions(
    mission_id: str | None = None,
    agent_id: str | None = None,
    intervention_type: str | None = None,
    limit: int = 100,
) -> str:
    """List supervisor interventions.

    Args:
        mission_id: Filter by mission.
        agent_id: Filter by agent.
        intervention_type: Filter by type.
        limit: Max results.
    """
    kwargs: dict[str, Any] = {"limit": limit}
    if mission_id:
        kwargs["mission_id"] = mission_id
    if agent_id:
        kwargs["agent_id"] = agent_id
    if intervention_type:
        kwargs["intervention_type"] = intervention_type
    return _call_backend_json("list_interventions", **kwargs)
