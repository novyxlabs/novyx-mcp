# Security Policy

## Supported Versions

Security fixes target the latest published `novyx-mcp` release on PyPI.

## Reporting A Vulnerability

Please report security issues privately through GitHub's private vulnerability
reporting flow when available, or email `security@novyxlabs.com`.

Include:

- affected `novyx-mcp` version
- MCP client and version
- Python version
- local mode or Novyx Cloud mode
- minimal reproduction steps
- whether secrets, tenant data, or action approval state may be exposed

Do not open public issues for vulnerabilities involving credentials, tenant
isolation, approval bypass, SSRF, arbitrary file access, or remote code
execution.

## Trust Boundaries

- Local SQLite mode stores memory state on the user's machine.
- Cloud-only tools require a Novyx API key and should fail explicitly when cloud
  capability is unavailable.
- Cross-tenant sharing must not claim success unless hosted invitation
  redemption exists and verifies the recipient.
- Tool responses must not invent URLs, tokens, approvals, or persisted objects
  that were not created by a backing system.
