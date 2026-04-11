# Changelog

All notable changes to `novyx-mcp` are documented here.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note on this repo**: This is a publishing mirror of the canonical `novyx-mcp` package, which lives in the Novyx Labs monorepo at `packages/novyx-mcp/`. The PyPI package (`pip install novyx-mcp`) is built and published from the monorepo, not from this repo. This repo exists for issue tracking, marketplace listings, and external visibility.
>
> If you find a discrepancy between the source code in this repo and the published PyPI package, the published package is authoritative.

## [Unreleased] — stale mirror sync

### Fixed
- `novyx_mcp/__init__.py` `__version__` had been stuck at `2.1.5` since the initial publishing-mirror sync. Bumped to `2.5.0` to match the current PyPI release.
- `pyproject.toml` `version` was at `2.4.0`, lagging the published package by one minor. Bumped to `2.5.0`.
- `pyproject.toml` and `mcp.json` description claims updated to match the published tool count (`119`).
- `mcp.json` `version` and `packages.pypi.version` bumped from `2.2.0` to `2.5.0`.

### Known stale (pending full re-sync from monorepo)
- `mcp.json` `tools` / `resources` / `prompts` arrays still reflect the v2.2.0 surface and need to be regenerated from the monorepo's `server.py` (which has 119 `@mcp.tool` decorators with `annotations=ToolAnnotations(...)`). Not done in this PR because the standalone repo's `novyx_mcp/server.py` is itself a stale mirror — regenerating from it would publish a manifest claiming 27 tools for a 119-tool package.
- `novyx_mcp/server.py` source itself is stale (27 decorators vs the monorepo's 119) and needs to be re-synced.

### Added
- This `CHANGELOG.md`.
- A note at the top of this file flagging that this repo is a publishing mirror, not the source of truth.

## [2.5.0] — 2026-04 (current PyPI)

Published from the monorepo. 119 MCP tools, full Runtime v2 surface.

## [2.4.0] — 2026-03-31

### Added
- Core memory works locally with zero config (SQLite). Cloud mode unlocks the full surface: threat intelligence, auto-defense, correlation, governed actions, Runtime v2 agents/missions/capabilities, cortex, replay, and eval baselines.
- Brand line: "Every Novyx customer makes every other customer safer."
- Badges: PyPI, License, MCP tools, Demo.

## [2.2.0]

### Added
- Phase 2 sentinel intel, auto-defense, and correlation tools.
- Sync with Novyx Core.

## [2.1.5] — initial release

- First public release of `novyx-mcp` standalone repo.
- Core memory primitives, knowledge graph, context spaces, replay, cortex, control.
- Local SQLite backend (zero config) and cloud backend via the Novyx Python SDK.
