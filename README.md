<!-- mcp-name: io.github.novyxlabs/novyx-mcp -->
# novyx-mcp

Persistent memory for AI agents. 23 MCP tools for **Claude Desktop**, **Cursor**, and **Claude Code**. Install the server, add a section to your `CLAUDE.md`, and every Claude Code session shares context — turning isolated agents into a coordinated team. Includes context spaces for multi-agent collaboration, replay for time-travel debugging, and cortex for autonomous memory intelligence.

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

### Core Memory (10 tools)

| Tool | Description |
|------|-------------|
| `remember` | Store a memory observation with optional tags, importance, context, and TTL |
| `recall` | Search memories semantically using natural language |
| `forget` | Delete a memory by UUID |
| `list_memories` | List stored memories with optional tag filtering |
| `memory_stats` | Get memory statistics (total count, average importance, etc.) |
| `rollback` | Rollback memory to a point in time (supports dry run) |
| `audit` | Get the audit trail of memory operations |
| `link_memories` | Create a directed link between two memories |
| `add_triple` | Add a knowledge graph triple (subject -> predicate -> object) |
| `query_triples` | Query knowledge graph triples with filters |

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

### Replay (4 tools — Pro+)

Time-travel debugging — inspect how memory changed over time.

| Tool | Description |
|------|-------------|
| `replay_timeline` | Get a chronological timeline of memory operations |
| `replay_snapshot` | Reconstruct memory state at a specific point in time |
| `replay_lifecycle` | Trace the full lifecycle of a single memory |
| `replay_diff` | Compare memory state between two points in time |

### Cortex (3 tools — Pro+)

Autonomous memory intelligence — consolidation, reinforcement, and insights.

| Tool | Description |
|------|-------------|
| `cortex_status` | Check cortex configuration and last run stats |
| `cortex_run` | Trigger a cortex cycle (consolidation + reinforcement) |
| `cortex_insights` | Get AI-generated insights from memory patterns (Enterprise) |

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
