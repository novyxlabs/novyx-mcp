<!-- mcp-name: io.github.novyxlabs/novyx-mcp -->
# novyx-mcp

Persistent memory for AI agents. 91 MCP tools for **Claude Desktop**, **Cursor**, and **Claude Code**. Works locally with zero config (SQLite) for core memory operations, or connects to Novyx Cloud for the full surface including threat intelligence, auto-defense, correlation, governed actions, cortex, replay, and eval baselines. Every Novyx customer makes every other customer safer.

## Install

```bash
pip install novyx-mcp
```

## Configuration

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "novyx-memory": {
      "command": "python",
      "args": ["-m", "novyx_mcp"],
      "env": {
        "NOVYX_API_KEY": "nram_your_key_here"
      }
    }
  }
}
```

### Cursor

Add to your Cursor MCP settings:

```json
{
  "mcpServers": {
    "novyx-memory": {
      "command": "python",
      "args": ["-m", "novyx_mcp"],
      "env": {
        "NOVYX_API_KEY": "nram_your_key_here"
      }
    }
  }
}
```

### Claude Code

```bash
claude mcp add novyx-memory -- python -m novyx_mcp
```

Set the `NOVYX_API_KEY` environment variable before starting Claude Code. Omit it to use local mode (SQLite, zero config).

### CLAUDE.md Integration

After installing and configuring the MCP server above, add this to your project's `CLAUDE.md` so Claude Code uses Novyx automatically:

```markdown
## Shared Memory (Novyx MCP)
You have access to novyx-mcp tools for shared memory, knowledge graph, audit,
rollback, replay, and context spaces. Use them when relevant. Store decisions
and status at the end of tasks. Check for context from other agents before
starting new work.
```

This turns isolated Claude Code sessions into a coordinated team — each session stores what it learned and checks what other sessions have done before starting work.

## Available Tools

### Core Memory (5 tools)

| Tool | Description |
|------|-------------|
| `remember` | Store a memory observation with optional tags, importance, context, and TTL |
| `recall` | Search memories semantically using natural language |
| `forget` | Delete a memory by UUID |
| `list_memories` | List stored memories with optional tag filtering |
| `memory_stats` | Get memory statistics (total count, average importance, etc.) |

### Draft Workflow (7 tools)

Stage, review, and merge memory changes before they go live.

| Tool | Description |
|------|-------------|
| `draft_memory` | Create a draft memory for review before committing |
| `memory_drafts` | List pending drafts, optionally filtered by status or branch |
| `draft_diff` | Compare a draft against existing memories |
| `merge_draft` | Approve and commit a draft into live memory |
| `reject_draft` | Reject a draft with an optional reason |
| `memory_branch` | View all drafts in a branch |
| `merge_branch` | Merge all approved drafts in a branch |
| `reject_branch` | Reject all drafts in a branch |

### Knowledge Graph (11 tools)

Store and query structured relationships between entities.

| Tool | Description |
|------|-------------|
| `add_triple` | Add a knowledge graph triple (subject -> predicate -> object) |
| `query_triples` | Query knowledge graph triples with filters |
| `delete_triple` | Delete a triple by ID |
| `link_memories` | Create a directed link between two memories |
| `unlink` | Remove a link between two memories |
| `get_links` | Get all links for a memory, optionally filtered by relation |
| `graph_edges` | Query graph edges with filtering |
| `list_entities` | List entities in the knowledge graph |
| `get_entity` | Get details for a specific entity |
| `delete_entity` | Delete an entity from the knowledge graph |
| `supersede` | Mark one memory as superseding another |

### Rollback & Audit (6 tools)

Cryptographic audit trail and point-in-time recovery.

| Tool | Description |
|------|-------------|
| `rollback` | Rollback memory to a point in time (supports dry run) |
| `rollback_preview` | Preview what a rollback would change before executing |
| `rollback_history` | View past rollback operations |
| `audit` | Get the audit trail of memory operations |
| `audit_verify` | Verify the integrity of the cryptographic audit chain |
| `audit_export` | Export the full audit log (JSON, CSV, or JSONL) |

### Context Spaces (6 tools)

Multi-agent collaboration — shared memory with fine-grained permissions.

| Tool | Description |
|------|-------------|
| `create_space` | Create a shared context space for multi-agent collaboration |
| `list_spaces` | List all context spaces you own or have access to |
| `space_memories` | Search or list memories within a specific context space |
| `update_space` | Update space settings (description, allowed agents, tags) |
| `delete_space` | Delete a context space and disassociate its memories |
| `share_space` | Share a context space with another user by email |

### Sharing (3 tools)

Share memory context across agents and users.

| Tool | Description |
|------|-------------|
| `accept_shared_context` | Accept a shared context invitation by token |
| `shared_contexts` | List all shared context tokens |
| `revoke_shared_context` | Revoke a shared context token |

### Replay (7 tools — Pro+)

Time-travel debugging — inspect how memory changed over time.

| Tool | Description |
|------|-------------|
| `replay_timeline` | Get a chronological timeline of memory operations |
| `replay_snapshot` | Reconstruct memory state at a specific point in time |
| `replay_lifecycle` | Trace the full lifecycle of a single memory |
| `replay_diff` | Compare memory state between two points in time |
| `replay_memory` | Replay the full history of a specific memory |
| `replay_recall` | Run a recall query against memory state at a past timestamp |
| `replay_memory_drift` | Measure how memory changed between two timestamps |

### Execution Tracing (4 tools)

Track multi-step agent workflows with cryptographic verification.

| Tool | Description |
|------|-------------|
| `trace_create` | Create a new execution trace |
| `trace_step` | Add a step to an active trace |
| `trace_complete` | Mark a trace as complete |
| `trace_verify` | Verify the integrity of a trace's step chain |

### Eval (7 tools)

Score and monitor memory quality over time. Includes baseline regression testing.

| Tool | Description |
|------|-------------|
| `eval_run` | Run a memory quality evaluation |
| `eval_gate` | Gate a workflow on a minimum memory quality score |
| `eval_history` | View past evaluation results |
| `eval_drift` | Measure memory drift over a time window |
| `eval_baseline_create` | Save a recall baseline for regression testing |
| `eval_baselines` | List all saved eval baselines |
| `eval_baseline_delete` | Delete an eval baseline |

### Cortex (5 tools — Pro+)

Autonomous memory intelligence — consolidation, reinforcement, and insights.

| Tool | Description |
|------|-------------|
| `cortex_status` | Check cortex configuration and last run stats |
| `cortex_run` | Trigger a cortex cycle (consolidation + reinforcement) |
| `cortex_insights` | Get AI-generated insights from memory patterns (Enterprise) |
| `cortex_config` | View cortex configuration details |
| `cortex_update_config` | Tune consolidation threshold, reinforcement boost, and decay rate |

### Control (7 tools)

Governed actions with policy evaluation and approval workflows.

| Tool | Description |
|------|-------------|
| `list_pending` | List actions awaiting approval |
| `approve_action` | Approve a pending action |
| `check_policy` | Check what policies apply to a connector/environment |
| `action_history` | View past action submissions and outcomes |
| `action_submit` | Submit an action for governed execution |
| `action_status` | Get the status of a specific action |
| `explain_action` | Get the full causal chain for why an action was blocked/approved |

### Utilities (3 tools)

| Tool | Description |
|------|-------------|
| `context_now` | Get current context (time, session, agent info) |
| `dashboard` | Get a full dashboard summary of memory state |
| `memory_health` | Check memory health score and diagnostics |

## Available Resources

| URI | Description |
|-----|-------------|
| `novyx://memories` | List all stored memories |
| `novyx://memories/{memory_id}` | Get a specific memory by UUID |
| `novyx://stats` | Memory statistics |
| `novyx://usage` | Usage and plan information |
| `novyx://spaces` | List all context spaces |
| `novyx://spaces/{space_id}` | Get a specific context space |

## Available Prompts

| Prompt | Description |
|--------|-------------|
| `memory-context` | Recall relevant memories and format them as context (takes a `query` argument) |
| `session-summary` | List all memories for a session (takes a `session_id` argument) |
| `space-context` | Recall memories from a specific context space (takes `space_id` and `query` arguments) |

## Get an API Key

Sign up at [novyxlabs.com](https://novyxlabs.com) to get your API key. The free tier includes 5,000 memories and 5,000 API calls per month.

## License

MIT - Copyright 2026 Novyx Labs
