# Changelog

## 2.5.1 (2026-04-11)

### Fixed — Runtime v2 MCP tools were non-functional in 2.5.0

The 29 Runtime v2 MCP tools (agents, missions, capabilities, checkpoints,
interventions) shipped in 2.5.0 dispatched through `_call_backend_json`
to 25 method names that did not exist on either `LocalBackend` or
`CloudBackend`. Every Runtime v2 tool call returned
`{"error": "'CloudBackend' object has no attribute '<method>'"}` to the
caller. The MCP surface was advertised as 119 tools but only 90 actually
worked.

Root cause: Runtime v2 was added to `server.py` without adding the
corresponding backend dispatch methods. The SDK had all 25 methods, so
Python-SDK consumers were unaffected — only the MCP path was broken.
Caught by Codex review on PR #6, not by tests, because the existing test
suite exercised backend methods directly and never went through the
MCP tool layer for Runtime v2.

**Fix**:
- `cloud_backend.py`: added 25 delegation methods covering agents,
  missions, capabilities, checkpoints, and interventions. Each method
  passes through to the corresponding `self._client.<method>()` on the
  Novyx Python SDK.
- `local_backend.py`: added 25 stub methods that raise `CloudFeatureError`
  with an upgrade prompt. Runtime v2 orchestration state lives on the
  Novyx Cloud API — local SQLite mode holds memory only.

### Added — Backend dispatch guardrail test

`tests/test_backend_dispatch.py` programmatically walks `server.py`,
extracts every `_call_backend_json("<method>", ...)` call, and asserts
each method is implemented on both `LocalBackend` and `CloudBackend`.
This is a regression test for exactly this class of bug. Any future
MCP tool that dispatches to a method name the backends don't implement
will fail CI immediately, listing the specific missing methods. The
same class of drift cannot ship again.

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
