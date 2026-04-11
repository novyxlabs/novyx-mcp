<!-- mcp-name: io.github.novyxlabs/novyx-mcp -->
# novyx-mcp

Persistent memory + governance for AI agents. 119 MCP tools for **Claude Desktop**, **Cursor**, and **Claude Code**. Works locally with zero config (SQLite) for core memory operations, or connects to Novyx Cloud for the full surface including policy-as-code, approval workflows, governance dashboard, Runtime v2 agents/missions/capabilities, threat intelligence, auto-defense, correlation, governed actions, cortex, replay, and eval baselines. Every Novyx customer makes every other customer safer.

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

## Canonical Workflow: Draft, Review, Merge

This is the highest-signal Novyx workflow for coding agents:

1. The agent learns something important, but uses `draft_memory(..., branch_id="feature-x")` instead of writing directly.
2. Review the whole branch with `memory_branch("feature-x")`.
3. Use `draft_diff` when one draft needs a closer look.
4. Merge the whole branch with `merge_branch("feature-x")`, or reject it with `reject_branch("feature-x")`.

Example:

```text
draft_memory(
  observation="Deploys fail if REDIS_URL is unset in staging",
  tags=["ops", "staging"],
  importance=8,
  branch_id="staging-fixes"
)

memory_branch("staging-fixes")

draft_diff("drf_abc123")

merge_branch("staging-fixes")
```

This keeps agent memory reviewable instead of letting every session write directly into permanent state.

## Available Tools

**119 tools across 11 categories.** Memory-only tools work in local SQLite mode (zero config, no API key). Cloud-only tools require a Novyx API key.

### Core Memory (19 tools)

Store, recall, supersede, and audit individual memories.

| Tool | Description |
|------|-------------|
| `remember` | Store a memory observation with tags, importance, context, TTL |
| `recall` | Semantic search using natural language |
| `list_memories` | List stored memories with optional tag filtering |
| `memory_stats` | Total count, average importance, conflict count |
| `memory_health` | Health score, stale memory count, contradiction count |
| `forget` | Delete a memory by UUID |
| `supersede` | Replace a memory with a new version, preserving history |
| `link_memories` | Create a directed link between two memories |
| `unlink` | Remove a link between memories |
| `get_links` | Retrieve all links for a memory |
| `add_triple` | Add a knowledge graph triple (subject → predicate → object) |
| `query_triples` | Query knowledge graph triples with filters |
| `delete_triple` | Remove a knowledge graph triple |
| `get_entity` / `list_entities` / `delete_entity` | Knowledge graph entity CRUD |
| `graph_edges` | List edges between memories or entities |
| `audit` | Get the cryptographic audit trail |
| `audit_export` / `audit_verify` | Export and verify the audit chain |

### Memory Drafts & Branches (8 tools)

Stage memory changes for review before committing.

| Tool | Description |
|------|-------------|
| `draft_memory` | Create a reviewable draft before writing to canonical memory |
| `memory_drafts` | List open, merged, or rejected drafts |
| `draft_diff` | Show field-level changes before merging a draft |
| `merge_draft` / `reject_draft` | Merge or reject an individual draft |
| `memory_branch` | Review a whole branch/session of drafts at once |
| `merge_branch` / `reject_branch` | Merge or reject every open draft in a branch |

### Rollback (3 tools)

Time-travel restore — undo agent mistakes.

| Tool | Description |
|------|-------------|
| `rollback` | Rollback memory to a point in time (supports dry run) |
| `rollback_preview` | Preview what a rollback would change |
| `rollback_history` | List all prior rollback operations |

### Context Spaces (8 tools)

Multi-agent collaboration — shared memory with fine-grained permissions.

| Tool | Description |
|------|-------------|
| `create_space` / `update_space` / `delete_space` | Context space CRUD |
| `list_spaces` | List spaces you own or have access to |
| `space_memories` | Search or list memories within a space |
| `share_space` | Share a space by email with permission level |
| `shared_contexts` | List spaces shared with you |
| `accept_shared_context` / `revoke_shared_context` | Accept invites or revoke access |
| `context_now` | Get the current context state for a space |

### Novyx Control — Governance (10 tools)

Policy-as-code, approval workflows, and governed actions. New in Phase 1-5 (v2.5.0).

| Tool | Description | Tier |
|------|-------------|------|
| `create_policy` | Create a custom YAML policy with regex rules and severities | Starter+ |
| `list_policies` | List all active policies (built-in + custom) | All |
| `delete_policy` | Disable a custom policy | Starter+ |
| `check_policy` | Check the current Control policy profile | All |
| `action_submit` | Submit an action for policy evaluation | All |
| `action_status` | Get the status of a submitted action | All |
| `action_history` | List recent governed actions | All |
| `explain_action` | Get the full causal chain for an action | All |
| `list_pending` | List actions awaiting human approval | All |
| `approve_action` | Approve or deny a pending action | All |

### Runtime v2 — Agent Orchestration (29 tools)

First-class agents, missions, capability packs, checkpoints, and human interventions.

| Category | Tools |
|----------|-------|
| **Agents** | `create_agent`, `get_agent`, `list_agents`, `update_agent`, `delete_agent` |
| **Missions** | `create_mission`, `get_mission`, `list_missions`, `update_mission`, `delete_mission`, `pause_mission`, `resume_mission`, `cancel_mission` |
| **Capabilities** | `create_capability`, `get_capability`, `list_capabilities`, `update_capability`, `delete_capability` |
| **Checkpoints** | `create_checkpoint`, `get_checkpoint`, `list_checkpoints`, `rollback_to_checkpoint` |
| **Interventions** | `create_intervention`, `get_intervention`, `list_interventions` |

Capabilities require Starter+. Checkpoints require Pro+. Interventions require Enterprise.

### Threat Intelligence (9 tools — Pro+)

Detect, signature, and correlate adversarial activity across agents.

| Tool | Description |
|------|-------------|
| `threat_feed` | Subscribe to the threat intelligence feed |
| `threat_record` | Log a threat observation |
| `threat_match` | Match an event against known signatures |
| `threat_signature` | Create or query a threat signature |
| `threat_mitigate` | Apply a mitigation for a known threat |
| `threat_trending` | Trending threats over time |
| `threat_stats` | Aggregate threat statistics |
| `correlate_threat` | Correlate a single event across the chain |
| `coordinated_attack_check` | Detect coordinated multi-agent attack patterns |
| `detect_campaign` | Detect long-running threat campaigns |
| `related_signatures` | Find signatures related to a given threat |

### Auto-Defense (7 tools — Pro+)

Deploy and tune automated defensive rules.

| Tool | Description |
|------|-------------|
| `defense_deploy` | Deploy a new defense rule |
| `defense_list` | List all active defenses |
| `defense_remove` | Remove a defense rule |
| `defense_recommend` | Get AI-recommended defenses for current threats |
| `defense_effectiveness` | Measure how effective a defense has been |
| `defense_record_block` | Log a successful block by a defense |
| `defense_stats` | Aggregate defense performance stats |

### Replay (6 tools — Pro+)

Time-travel debugging — inspect how memory changed over time.

| Tool | Description |
|------|-------------|
| `replay_timeline` | Chronological timeline of memory operations |
| `replay_snapshot` | Reconstruct memory state at a point in time |
| `replay_lifecycle` | Trace the full lifecycle of a single memory |
| `replay_diff` | Compare memory state between two points |
| `replay_memory` | Replay a single memory's history |
| `replay_memory_drift` | Show how a memory drifted over time |
| `replay_recall` | Replay a recall query as it would have answered then |

### Eval (7 tools)

Memory health evaluation and CI/CD gates.

| Tool | Description | Tier |
|------|-------------|------|
| `eval_run` | Run a full memory health evaluation | All |
| `eval_history` | Get historical eval scores | All |
| `eval_drift` | Detect drift since the last baseline | All |
| `eval_gate` | Pass/fail gate for CI/CD pipelines | Pro+ |
| `eval_baseline_create` / `eval_baselines` / `eval_baseline_delete` | Baseline CRUD |

### Cortex (4 tools — Pro+)

Autonomous memory intelligence — consolidation, reinforcement, and insights.

| Tool | Description |
|------|-------------|
| `cortex_status` | Check cortex configuration and last run stats |
| `cortex_config` / `cortex_update_config` | Get/update cortex configuration |
| `cortex_run` | Trigger a cortex cycle (consolidation + reinforcement) |
| `cortex_insights` | Get AI-generated insights from memory patterns (Enterprise) |

### Traces (4 tools)

Sentinel trace logging for full agent step audits.

| Tool | Description |
|------|-------------|
| `trace_create` | Start a new trace |
| `trace_step` | Append a step to a trace |
| `trace_complete` | Finalize a trace |
| `trace_verify` | Cryptographically verify a trace |

### Operational (4 tools)

| Tool | Description |
|------|-------------|
| `dashboard` | Aggregated stats — usage, pressure, governance counts |
| `stream_status` | Status of any active streams |

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

Sign up at [novyxlabs.com](https://novyxlabs.com) to get your API key. The free tier includes 5,000 memories and 5,000 API calls per month, including the draft-review-merge workflow.

## License

MIT - Copyright 2026 Novyx Labs
