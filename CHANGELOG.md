# Changelog

## 2.5.0 (2026-04-10)

### Added — Phase 1: Policy-as-Code MCP Tools

- `create_policy` — Define a custom YAML policy with regex rules,
  severities, and per-rule outcomes (`block | require_approval | warn`).
  Requires Starter plan or above.
- `list_policies` — List all active policies (built-in + tenant-defined
  custom). Available on all tiers.
- `delete_policy` — Disable a custom policy. Built-in policies cannot
  be deleted.

### Changed — Phase 3: Multi-Provider Neutrality (BREAKING)

- `create_agent` tool now requires `provider` and `model` parameters.
  Returns an error if either is missing or if `provider` is not one of
  `"openai"`, `"anthropic"`, `"litellm"`. Use `litellm` for any model not
  directly supported (Gemini, Mistral, Cohere, Ollama, etc.).

### Fixed

- Version sync: `__init__.py` was stuck at 2.1.5 while `pyproject.toml`
  had advanced. Both now correctly report 2.5.0.
- Updated package description to reflect 119 total tools (was 107) and
  call out policy-as-code + governance dashboard surfaces.

### Tool count

119 tools across 11 categories: Core Memory, Drafts & Branches, Rollback,
Context Spaces, Novyx Control (Governance), Runtime v2, Threat
Intelligence, Auto-Defense, Replay, Eval, Cortex, Traces, Operational.
See README for the full list.

## 2.4.0 (2026-03-XX)

Runtime v2 MCP tools — agents, missions, capabilities, checkpoints,
interventions (29 new tools).

## 2.3.0

- Bumped to 2.3.0, published to PyPI.

## 2.1.5 (2026-03-XX)

- Local-first SQLite mode (no API key required for memory operations).
- CLAUDE.md descriptions on all tools for better Claude Code DX.

## 1.0.0 (2026-02-24)

Initial release.

- 10 MCP tools: remember, recall, forget, list_memories, memory_stats, rollback, audit, link_memories, add_triple, query_triples
- 4 MCP resources: novyx://memories, novyx://memories/{id}, novyx://stats, novyx://usage
- 2 MCP prompts: memory-context, session-summary
- Works with Claude Desktop, Cursor, and Claude Code
