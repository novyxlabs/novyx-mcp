"""Machine-readable registry of every MCP tool exposed by this server.

Single source of truth for tool-surface tests, tool_health introspection,
and the CI parity gate that blocks new tools from shipping without a
registry entry. See test_mcp_tool_surface.py.

Status legend:
  - "functional"            : calls the backend via SDK or _call_backend_json.
                              Exercised end-to-end by test_mcp_tool_surface.
  - "cloud_only"            : routes through _control_request and raises a
                              structured CloudFeatureError when the Control
                              service is not configured.
  - "cloud_only_hard_fail"  : reserved. A cloud-only tool that does NOT emit
                              a structured envelope on config-missing. Any
                              tool landing here is a regression bug.
  - "stub"                  : reserved. Tool is declared but raises
                              NotImplementedError or returns a canned stub.
"""

from __future__ import annotations

from typing import Literal, TypedDict

ToolStatus = Literal["functional", "cloud_only", "cloud_only_hard_fail", "stub"]


class ToolInfo(TypedDict):
    """Registry entry for a single MCP tool."""

    status: ToolStatus
    category: str
    description: str


TOOL_REGISTRY: dict[str, ToolInfo] = {
    # --- audit ---
    "audit": {"status": "functional", "category": "audit", "description": "Get the audit trail of memory operations."},
    "audit_export": {"status": "functional", "category": "audit", "description": "Export the full audit log."},
    "audit_verify": {"status": "functional", "category": "audit", "description": "Verify the integrity of the audit trail."},

    # --- control ---
    "action_history": {"status": "cloud_only", "category": "control", "description": "List recent Control actions with their status."},
    "action_status": {"status": "cloud_only", "category": "control", "description": "Get the status of a specific Control action."},
    "action_submit": {"status": "cloud_only", "category": "control", "description": "Submit an action to Novyx Control for governed execution."},
    "approve_action": {"status": "cloud_only", "category": "control", "description": "Approve a pending agent action in Novyx Control."},
    "check_policy": {"status": "cloud_only", "category": "control", "description": "Check the current Control policy profile."},
    "create_policy": {"status": "cloud_only", "category": "control", "description": "Create a custom governance policy for your agent's actions."},
    "delete_policy": {"status": "cloud_only", "category": "control", "description": "Disable a custom governance policy."},
    "list_pending": {"status": "cloud_only", "category": "control", "description": "List pending Control approval requests."},
    "list_policies": {"status": "cloud_only", "category": "control", "description": "List all active Control policies (built-in + custom)."},

    # --- cortex ---
    "cortex_config": {"status": "functional", "category": "cortex", "description": "Get the current Cortex configuration."},
    "cortex_insights": {"status": "functional", "category": "cortex", "description": "List auto-generated memory insights."},
    "cortex_run": {"status": "functional", "category": "cortex", "description": "Manually trigger a Cortex cycle."},
    "cortex_status": {"status": "functional", "category": "cortex", "description": "Get Cortex autonomous intelligence status."},
    "cortex_update_config": {"status": "functional", "category": "cortex", "description": "Update Cortex configuration."},

    # --- defense ---
    "defense_deploy": {"status": "functional", "category": "defense", "description": "Deploy a defense rule against a threat signature."},
    "defense_effectiveness": {"status": "functional", "category": "defense", "description": "Measure the effectiveness of a deployed defense."},
    "defense_list": {"status": "functional", "category": "defense", "description": "List active auto-deployed defense rules."},
    "defense_recommend": {"status": "functional", "category": "defense", "description": "Get a recommended defense strategy for a threat signature."},
    "defense_record_block": {"status": "functional", "category": "defense", "description": "Record that a defense blocked a threat."},
    "defense_remove": {"status": "functional", "category": "defense", "description": "Remove a deployed defense rule."},
    "defense_stats": {"status": "functional", "category": "defense", "description": "Get overall defense statistics."},

    # --- eval ---
    "eval_baseline_create": {"status": "functional", "category": "eval", "description": "Save a recall baseline for regression testing."},
    "eval_baseline_delete": {"status": "functional", "category": "eval", "description": "Delete an eval baseline."},
    "eval_baselines": {"status": "functional", "category": "eval", "description": "List all saved eval baselines."},
    "eval_drift": {"status": "functional", "category": "eval", "description": "Detect memory drift over a time period."},
    "eval_gate": {"status": "functional", "category": "eval", "description": "CI gate — pass or fail based on memory health score."},
    "eval_history": {"status": "functional", "category": "eval", "description": "List past memory evaluation runs."},
    "eval_run": {"status": "functional", "category": "eval", "description": "Run a memory health evaluation."},

    # --- introspection ---
    "tool_health": {"status": "functional", "category": "introspection", "description": "Introspect the MCP tool surface — status, category, description per tool."},

    # --- graph ---
    "add_triple": {"status": "functional", "category": "graph", "description": "Add a knowledge graph triple (subject -> predicate -> object)."},
    "delete_entity": {"status": "functional", "category": "graph", "description": "Delete a knowledge graph entity and all its triples."},
    "delete_triple": {"status": "functional", "category": "graph", "description": "Delete a knowledge graph triple."},
    "get_entity": {"status": "functional", "category": "graph", "description": "Get a knowledge graph entity and its associated triples."},
    "get_links": {"status": "functional", "category": "graph", "description": "Get all links for a memory."},
    "graph_edges": {"status": "functional", "category": "graph", "description": "Query the memory graph edges with filters."},
    "link_memories": {"status": "functional", "category": "graph", "description": "Create a directed link between two memories."},
    "list_entities": {"status": "functional", "category": "graph", "description": "List knowledge graph entities."},
    "query_triples": {"status": "functional", "category": "graph", "description": "Query knowledge graph triples with optional filters."},
    "unlink": {"status": "functional", "category": "graph", "description": "Remove a link between two memories."},

    # --- memory ---
    "context_now": {"status": "functional", "category": "memory", "description": "Get a snapshot of your current memory context."},
    "draft_diff": {"status": "functional", "category": "memory", "description": "Show a field-level diff for a memory draft."},
    "draft_memory": {"status": "functional", "category": "memory", "description": "Create a reviewable draft without writing to canonical memory."},
    "forget": {"status": "functional", "category": "memory", "description": "Delete a memory by its UUID."},
    "list_memories": {"status": "functional", "category": "memory", "description": "List stored memories with optional tag filtering."},
    "memory_branch": {"status": "functional", "category": "memory", "description": "Get grouped review information for a branch/session of drafts."},
    "memory_drafts": {"status": "functional", "category": "memory", "description": "List current memory drafts."},
    "memory_health": {"status": "functional", "category": "memory", "description": "Check the health of your agent's memory."},
    "memory_stats": {"status": "functional", "category": "memory", "description": "Get memory statistics for the current account."},
    "merge_branch": {"status": "functional", "category": "memory", "description": "Merge all open drafts in a branch/session."},
    "merge_draft": {"status": "functional", "category": "memory", "description": "Merge a reviewed draft into canonical memory."},
    "recall": {"status": "functional", "category": "memory", "description": "Search memories semantically using natural language."},
    "reject_branch": {"status": "functional", "category": "memory", "description": "Reject all open drafts in a branch/session."},
    "reject_draft": {"status": "functional", "category": "memory", "description": "Reject a draft without creating a memory."},
    "remember": {"status": "functional", "category": "memory", "description": "Store a memory observation in Novyx."},
    "supersede": {"status": "functional", "category": "memory", "description": "Mark a memory as superseded by a newer one."},

    # --- replay ---
    "replay_diff": {"status": "functional", "category": "replay", "description": "Diff memory state between two timestamps."},
    "replay_lifecycle": {"status": "functional", "category": "replay", "description": "Full biography of a single memory."},
    "replay_memory": {"status": "functional", "category": "replay", "description": "Get the full history of a single memory."},
    "replay_memory_drift": {"status": "functional", "category": "replay", "description": "Detect memory drift between two timestamps."},
    "replay_recall": {"status": "functional", "category": "replay", "description": "Time-travel recall — what would search have returned at a past timestamp?"},
    "replay_snapshot": {"status": "functional", "category": "replay", "description": "Reconstruct memory state at a specific point in time."},
    "replay_timeline": {"status": "functional", "category": "replay", "description": "Get the full timeline of memory operations. The tape you scrub through."},

    # --- rollback ---
    "rollback": {"status": "functional", "category": "rollback", "description": "Rollback memory to a point in time."},
    "rollback_history": {"status": "functional", "category": "rollback", "description": "List past rollback operations."},
    "rollback_preview": {"status": "functional", "category": "rollback", "description": "Preview what a rollback would do without executing it."},

    # --- runtime ---
    "cancel_mission": {"status": "functional", "category": "runtime", "description": "Cancel a mission."},
    "create_agent": {"status": "functional", "category": "runtime", "description": "Create a first-class agent entity in the Novyx Runtime."},
    "create_capability": {"status": "functional", "category": "runtime", "description": "Register a capability pack (tool bundle with governance)."},
    "create_checkpoint": {"status": "functional", "category": "runtime", "description": "Create a checkpoint for a mission (rollback point)."},
    "create_intervention": {"status": "functional", "category": "runtime", "description": "Record a supervisor intervention (approve, reject, pause, escalate, etc)."},
    "create_mission": {"status": "functional", "category": "runtime", "description": "Create a mission (bounded job) for an agent."},
    "delete_agent": {"status": "functional", "category": "runtime", "description": "Delete an agent."},
    "delete_capability": {"status": "functional", "category": "runtime", "description": "Delete a capability pack."},
    "delete_mission": {"status": "functional", "category": "runtime", "description": "Delete a mission."},
    "get_agent": {"status": "functional", "category": "runtime", "description": "Get an agent by ID."},
    "get_capability": {"status": "functional", "category": "runtime", "description": "Get a capability pack by ID."},
    "get_checkpoint": {"status": "functional", "category": "runtime", "description": "Get a checkpoint by ID."},
    "get_intervention": {"status": "functional", "category": "runtime", "description": "Get a supervisor intervention by ID."},
    "get_mission": {"status": "functional", "category": "runtime", "description": "Get a mission by ID."},
    "list_agents": {"status": "functional", "category": "runtime", "description": "List all agents for the current tenant."},
    "list_capabilities": {"status": "functional", "category": "runtime", "description": "List registered capability packs."},
    "list_checkpoints": {"status": "functional", "category": "runtime", "description": "List checkpoints for a mission."},
    "list_interventions": {"status": "functional", "category": "runtime", "description": "List supervisor interventions."},
    "list_missions": {"status": "functional", "category": "runtime", "description": "List missions for the current tenant."},
    "pause_mission": {"status": "functional", "category": "runtime", "description": "Pause a running mission."},
    "resume_mission": {"status": "functional", "category": "runtime", "description": "Resume a paused mission."},
    "rollback_to_checkpoint": {"status": "functional", "category": "runtime", "description": "Rollback a mission to a previous checkpoint."},
    "update_agent": {"status": "functional", "category": "runtime", "description": "Update an existing agent's configuration."},
    "update_capability": {"status": "functional", "category": "runtime", "description": "Update an existing capability pack."},
    "update_mission": {"status": "functional", "category": "runtime", "description": "Update an existing mission's configuration."},

    # --- space ---
    "accept_shared_context": {"status": "functional", "category": "space", "description": "Accept a shared context invitation."},
    "create_space": {"status": "functional", "category": "space", "description": "Create a shared context space for multi-agent collaboration."},
    "delete_space": {"status": "functional", "category": "space", "description": "Delete a context space (owner only)."},
    "list_spaces": {"status": "functional", "category": "space", "description": "List all context spaces you can access."},
    "revoke_shared_context": {"status": "functional", "category": "space", "description": "Revoke a shared context invitation."},
    "share_space": {"status": "functional", "category": "space", "description": "Share a space/tag with another user by email."},
    "shared_contexts": {"status": "functional", "category": "space", "description": "List all shared contexts you have access to."},
    "space_memories": {"status": "functional", "category": "space", "description": "List or search memories within a context space."},
    "update_space": {"status": "functional", "category": "space", "description": "Update a context space (owner only)."},

    # --- system ---
    "dashboard": {"status": "functional", "category": "system", "description": "Get a full dashboard overview."},
    "stream_status": {"status": "functional", "category": "system", "description": "Get real-time memory stream connection status."},

    # --- threat ---
    "coordinated_attack_check": {"status": "functional", "category": "threat", "description": "Check if multiple threat events are part of a coordinated attack."},
    "correlate_threat": {"status": "functional", "category": "threat", "description": "Check if a threat correlates with attacks on other tenants."},
    "detect_campaign": {"status": "functional", "category": "threat", "description": "Detect an ongoing attack campaign."},
    "related_signatures": {"status": "functional", "category": "threat", "description": "Find threat signatures related to a given signature."},
    "threat_feed": {"status": "functional", "category": "threat", "description": "Get the anonymized threat intelligence feed."},
    "threat_match": {"status": "functional", "category": "threat", "description": "Find known threat signatures matching a threat event."},
    "threat_mitigate": {"status": "functional", "category": "threat", "description": "Mark a threat signature as mitigated."},
    "threat_record": {"status": "functional", "category": "threat", "description": "Record a threat event for cross-tenant intelligence."},
    "threat_signature": {"status": "functional", "category": "threat", "description": "Get a specific threat signature by ID."},
    "threat_stats": {"status": "functional", "category": "threat", "description": "Get overall threat intelligence statistics."},
    "threat_trending": {"status": "functional", "category": "threat", "description": "Get trending threat signatures."},

    # --- trace ---
    "explain_action": {"status": "functional", "category": "trace", "description": "Get the full causal chain for a Control action."},
    "trace_complete": {"status": "functional", "category": "trace", "description": "Mark an execution trace as complete."},
    "trace_create": {"status": "functional", "category": "trace", "description": "Create an execution trace to track a multi-step agent workflow."},
    "trace_step": {"status": "functional", "category": "trace", "description": "Add a step to an execution trace."},
    "trace_verify": {"status": "functional", "category": "trace", "description": "Verify an execution trace's integrity."},
}


def registry_snapshot() -> dict:
    """Return a JSON-serializable snapshot of the registry.

    Shape:
        {
            "counts": {"functional": int, "cloud_only": int, ...,
                       "total": int, "by_category": {...}},
            "tools":  [{"name": str, "status": str,
                        "category": str, "description": str}, ...],
        }
    """
    from collections import Counter

    status_counts = Counter(info["status"] for info in TOOL_REGISTRY.values())
    cat_counts = Counter(info["category"] for info in TOOL_REGISTRY.values())
    tools = [
        {
            "name": name,
            "status": info["status"],
            "category": info["category"],
            "description": info["description"],
        }
        for name, info in sorted(TOOL_REGISTRY.items())
    ]
    return {
        "counts": {
            **dict(status_counts),
            "total": len(TOOL_REGISTRY),
            "by_category": dict(cat_counts),
        },
        "tools": tools,
    }
