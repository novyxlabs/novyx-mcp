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


def _handle_tier_error(e: Exception, feature: str = "This feature") -> str:
    """Return a friendly upgrade message for tier-gated or cloud-only features."""
    from .local_backend import CloudFeatureError
    if isinstance(e, CloudFeatureError):
        return json.dumps({
            "error": str(e),
            "upgrade": "https://novyxlabs.com/pricing",
        })
    error_str = str(e)
    if "403" in error_str or "feature" in error_str.lower() or "Forbidden" in error_str:
        return json.dumps({
            "error": f"{feature} requires Pro tier or higher.",
            "upgrade": "https://novyxlabs.com/pricing",
        })
    return json.dumps({"error": error_str})


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
        kwargs: dict[str, Any] = {
            "observation": observation,
            "importance": importance,
        }
        if tags is not None:
            kwargs["tags"] = tags
        if context is not None:
            kwargs["context"] = context
        if ttl_seconds is not None:
            kwargs["ttl_seconds"] = ttl_seconds
        result = backend.remember(**kwargs)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def recall(
    query: str,
    limit: int = 5,
    tags: list[str] | None = None,
    min_score: float = 0.0,
) -> str:
    """Search memories semantically using natural language.

    Args:
        query: Natural language search query.
        limit: Maximum number of results to return (default 5).
        tags: Optional tag filter.
        min_score: Minimum similarity score 0-1 (default 0).

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
            return json.dumps(result, default=str)

        # Cloud mode: convert SearchResult to dict
        memories = []
        for mem in result.memories:
            memories.append({
                "uuid": mem.uuid,
                "observation": mem.observation,
                "tags": mem.tags,
                "importance": mem.importance,
                "score": mem.score,
                "created_at": mem.created_at,
            })
        return json.dumps({
            "query": result.query,
            "total_results": result.total_results,
            "memories": memories,
        }, default=str)
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


# =========================================================================
# Control Tools (requires NOVYX_CONTROL_URL + NOVYX_CONTROL_API_KEY)
# =========================================================================


def _control_request(method: str, path: str, body: dict | None = None) -> dict:
    """Make an HTTP request to Novyx Control."""
    import urllib.request
    import urllib.error

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
                f"{i}. **{obs}**\n"
                f"   - Tags: {tags_str} | Created: {mem.get('created_at', '')}\n"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error querying space: {e}"
