# Contributing To novyx-mcp

Thanks for helping improve Novyx MCP. This repository is the public publishing
mirror for the MCP package; the canonical source lives in
`novyxlabs/novyx-core/packages/novyx-mcp`.

## Development Flow

1. Open changes against `novyxlabs/novyx-core` when they touch package source,
   tests, README content, or package metadata.
2. The standalone `novyxlabs/novyx-mcp` mirror is generated from the monorepo.
3. Mirror-only changes should be limited to GitHub repository operations,
   workflow maintenance, and directory/listing metadata.

## Local Checks

From the monorepo root:

```bash
python3.11 -m pytest -q packages/novyx-mcp/tests
PYTHON=python3.11 ./scripts/test_all.sh --quick
```

For MCP surface changes, keep the registry and server decorators aligned. The
version consistency and tool surface tests exist to prevent public drift.

## What We Want

- Better Claude Desktop, Claude Code, Cursor, and Windsurf install paths.
- Honest local-vs-cloud capability descriptions.
- Small, reproducible demos for reviewable memory and governed actions.
- Bug reports with exact client config, Python version, and command output.

## What We Avoid

- Marketing claims that the local SQLite mode cannot satisfy.
- New tools without registry coverage and tool-surface tests.
- Mirror-only source edits that will be overwritten by the next monorepo sync.
