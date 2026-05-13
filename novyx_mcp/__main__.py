import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

NOVYX_DIR = Path("~/.novyx").expanduser()
CREDENTIALS_FILE = NOVYX_DIR / "credentials.json"
API_BASE = "https://novyx-ram-api.fly.dev"


def _load_stored_key():
    """Read API key from ~/.novyx/credentials.json."""
    try:
        data = json.loads(CREDENTIALS_FILE.read_text())
        return data.get("api_key")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def _save_stored_key(api_key, email):
    """Write API key to ~/.novyx/credentials.json (owner-only permissions)."""
    NOVYX_DIR.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_FILE.write_text(json.dumps({"api_key": api_key, "email": email}, indent=2) + "\n")
    try:
        CREDENTIALS_FILE.chmod(0o600)
    except OSError:
        pass  # Windows


def setup():
    """Get a free API key with one email input."""
    print("Novyx MCP Setup")
    print("=" * 40)

    existing = _load_stored_key()
    if existing:
        print(f"Already configured (key ending ...{existing[-6:]})")
        answer = input("Replace with a new key? [y/N]: ").strip().lower()
        if answer != "y":
            print("Setup cancelled.")
            return

    email = input("Email (for your free API key): ").strip()
    if not email or "@" not in email:
        print("Invalid email.", file=sys.stderr)
        sys.exit(1)

    print("Creating key...", end=" ", flush=True)
    payload = json.dumps({"email": email}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/v1/keys/cli",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
            # API may return message at top level or nested under detail
            detail = body.get("detail", body)
            if isinstance(detail, dict):
                msg = detail.get("message", detail.get("error", str(detail)))
            else:
                msg = str(detail)
        except Exception:
            msg = str(e)
        print(f"\nError: {msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nConnection error: {e}", file=sys.stderr)
        sys.exit(1)

    api_key = data.get("api_key")
    if not api_key:
        print("\nError: no key returned. Visit novyxlabs.com to create one manually.", file=sys.stderr)
        sys.exit(1)

    _save_stored_key(api_key, email)

    print("done.")
    print(f"\nKey saved to {CREDENTIALS_FILE}")
    print(f"Check {email} to verify (verification unlocks upgrades and key rotation).")
    print("\nFree tier active (5K memories, 5K calls/month). Restart your MCP client to connect.")
    print("Some features require Starter/Pro tier — see novyxlabs.com/pricing.")


def main():
    parser = argparse.ArgumentParser(description="Novyx MCP Server")
    parser.add_argument("--setup", action="store_true", help="Get a free API key")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio, use streamable-http for Managed Agents)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP transport (default: 8080)")
    args, _unknown = parser.parse_known_args()

    if args.setup:
        setup()
        return

    from novyx_mcp.server import mcp, startup_health_check
    startup_health_check()

    if args.transport == "streamable-http":
        import os
        os.environ.setdefault("FASTMCP_PORT", str(args.port))
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
