"""User-facing CLI and guided MCP client setup for Advassess."""
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP = "advassess"
CLIENTS = ("claude-code", "codex", "cursor", "gemini-cli", "claude-desktop")
LABELS = {
    "claude-code": "Claude Code",
    "codex": "Codex",
    "cursor": "Cursor",
    "gemini-cli": "Gemini CLI",
    "claude-desktop": "Claude Desktop",
}


def _home() -> Path:
    return Path.home()


def _config_path(client: str) -> Path:
    home = _home()
    if client == "claude-code":
        return home / ".claude.json"
    if client == "codex":
        return home / ".codex" / "config.toml"
    if client == "cursor":
        return home / ".cursor" / "mcp.json"
    if client == "gemini-cli":
        return home / ".gemini" / "settings.json"
    if client == "claude-desktop":
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
            return base / "Claude" / "claude_desktop_config.json"
        if sys.platform == "darwin":
            return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        return home / ".config" / "Claude" / "claude_desktop_config.json"
    raise ValueError(f"Unsupported client: {client}")


def _server_entry(workspace: Path) -> dict[str, Any]:
    # Running the installed package's Python interpreter avoids PATH and pipx shim issues.
    return {
        "command": sys.executable,
        "args": ["-m", "advassess.cli", "mcp", "--workspace", str(workspace)],
        "env": {},
    }


def _json_transform(path: Path, entry: dict[str, Any]) -> tuple[str, str]:
    original = path.read_text(encoding="utf-8") if path.exists() else "{}\n"
    try:
        data = json.loads(original or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse {path}: {exc}. No changes were made.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}. No changes were made.")
    data.setdefault("mcpServers", {})
    if not isinstance(data["mcpServers"], dict):
        raise ValueError(f"mcpServers is not an object in {path}. No changes were made.")
    data["mcpServers"][APP] = entry
    rendered = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    return original, rendered


def _toml_transform(path: Path, entry: dict[str, Any]) -> tuple[str, str]:
    try:
        import tomlkit
    except ImportError as exc:
        raise ValueError("Codex setup needs tomlkit. Reinstall Advassess with its declared dependencies.") from exc
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        doc = tomlkit.parse(original)
    except Exception as exc:
        raise ValueError(f"Cannot parse {path}: {exc}. No changes were made.") from exc
    servers = doc.get("mcp_servers")
    if servers is None:
        servers = tomlkit.table()
        doc["mcp_servers"] = servers
    if not hasattr(servers, "__setitem__"):
        raise ValueError(f"mcp_servers is not a TOML table in {path}. No changes were made.")
    table = tomlkit.table()
    table.add("command", entry["command"])
    table.add("args", entry["args"])
    if entry.get("env"):
        table.add("env", entry["env"])
    servers[APP] = table
    return original, tomlkit.dumps(doc)


def _transform(client: str, path: Path, entry: dict[str, Any]) -> tuple[str, str]:
    if client == "codex":
        return _toml_transform(path, entry)
    return _json_transform(path, entry)


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.advassess-backup-{stamp}")
    # Avoid overwriting a backup in the unlikely event of same-second repeated setup.
    suffix = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.advassess-backup-{stamp}-{suffix}")
        suffix += 1
    shutil.copy2(path, backup)
    return backup


def _confirm(prompt: str, yes: bool = False) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        raise ValueError("Interactive approval is required. Run in a terminal, or pass --yes after reviewing the plan.")
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _choose_clients(raw: str | None) -> list[str]:
    if raw:
        values = [v.strip().lower().replace("_", "-") for v in raw.split(",") if v.strip()]
        aliases = {"claude": "claude-code", "gemini": "gemini-cli", "desktop": "claude-desktop"}
        values = [aliases.get(v, v) for v in values]
        invalid = [v for v in values if v not in CLIENTS]
        if invalid:
            raise ValueError(f"Unknown client(s): {', '.join(invalid)}. Choose: {', '.join(CLIENTS)}")
        return list(dict.fromkeys(values))
    if not sys.stdin.isatty():
        raise ValueError("Specify --clients claude-code,codex,cursor,gemini-cli,claude-desktop or run setup interactively.")
    print("Choose agent clients to configure (comma-separated numbers; Enter selects all):")
    for i, key in enumerate(CLIENTS, 1):
        print(f"  {i}. {LABELS[key]}")
    raw_choice = input("Clients [1,2,3,4,5]: ").strip()
    if not raw_choice:
        return list(CLIENTS)
    try:
        numbers = [int(x.strip()) for x in raw_choice.split(",")]
        if any(n < 1 or n > len(CLIENTS) for n in numbers):
            raise ValueError("Client numbers must be between 1 and 5.")
        selected = [CLIENTS[n - 1] for n in numbers]
    except (ValueError, IndexError) as exc:
        raise ValueError("Invalid client selection.") from exc
    return list(dict.fromkeys(selected))


def setup(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists() or not workspace.is_dir():
        raise ValueError(f"Workspace must be an existing directory: {workspace}")
    clients = _choose_clients(args.clients)
    plans: list[tuple[str, Path, str, str]] = []
    entry = _server_entry(workspace)
    for client in clients:
        path = _config_path(client)
        before, after = _transform(client, path, entry)
        plans.append((client, path, before, after))

    print("\nAdvassess setup plan")
    print(f"Workspace: {workspace}")
    print(f"Python runtime: {sys.executable}")
    print("No files have been changed yet. Proposed configuration changes:")
    for client, path, before, after in plans:
        print(f"\n--- {LABELS[client]}: {path} ---")
        diff = difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile="current", tofile="proposed", lineterm="")
        shown = False
        for line in diff:
            print(line)
            shown = True
        if not shown:
            print("(no changes)")
    if not _confirm("Apply these changes? Existing files will be backed up first.", args.yes):
        print("Setup cancelled; no changes made.")
        return 1

    applied: list[tuple[Path, Path | None]] = []
    try:
        for _, path, _, after in plans:
            path.parent.mkdir(parents=True, exist_ok=True)
            backup = _backup(path)
            # Exclusive temp file in the same directory, then atomic replace.
            temp = path.with_name(path.name + f".advassess-tmp-{os.getpid()}")
            temp.write_text(after, encoding="utf-8")
            try:
                os.chmod(temp, 0o600)
            except OSError:
                pass
            os.replace(temp, path)
            applied.append((path, backup))
    except Exception:
        # Best-effort rollback: restore backups, or remove newly created config files.
        for path, backup in reversed(applied):
            if backup and backup.exists():
                shutil.copy2(backup, path)
            else:
                path.unlink(missing_ok=True)
        raise
    print("\nSetup complete. Restart the selected agent(s) to load the Advassess MCP server.")
    for client, path, _, _ in plans:
        print(f"  {LABELS[client]}: {path}")
    print("Existing configuration backups are named *.advassess-backup-<UTC timestamp>.")
    return 0


def uninstall(args: argparse.Namespace) -> int:
    clients = _choose_clients(args.clients)
    plans: list[tuple[str, Path, str, str]] = []
    for client in clients:
        path = _config_path(client)
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        if client == "codex":
            import tomlkit
            doc = tomlkit.parse(original)
            if "mcp_servers" in doc and APP in doc["mcp_servers"]:
                del doc["mcp_servers"][APP]
                if not doc["mcp_servers"]:
                    del doc["mcp_servers"]
            rendered = tomlkit.dumps(doc)
        else:
            data = json.loads(original)
            if isinstance(data, dict) and isinstance(data.get("mcpServers"), dict):
                data["mcpServers"].pop(APP, None)
                if not data["mcpServers"]:
                    data.pop("mcpServers", None)
            rendered = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        if rendered != original:
            plans.append((client, path, original, rendered))
    if not plans:
        print("No Advassess MCP entries found for the selected clients.")
        return 0
    print("The following Advassess MCP entries will be removed (other settings are preserved):")
    for client, path, _, _ in plans:
        print(f"  {LABELS[client]}: {path}")
    if not _confirm("Remove these entries? Backups will be created first.", args.yes):
        print("Uninstall cancelled; no changes made.")
        return 1
    for _, path, _, rendered in plans:
        _backup(path)
        temp = path.with_name(path.name + f".advassess-tmp-{os.getpid()}")
        temp.write_text(rendered, encoding="utf-8")
        os.replace(temp, path)
    print("Removed Advassess MCP entries. Restart your agent(s). The Python package remains installed.")
    return 0


def doctor(_: argparse.Namespace) -> int:
    print(f"Advassess CLI: {__file__}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    checks = []
    for mod, label in (("yaml", "PyYAML"), ("numpy", "NumPy"), ("sklearn", "scikit-learn"), ("mcp", "MCP SDK"), ("tomlkit", "TOML config support")):
        try:
            __import__(mod)
            checks.append((label, True, "available"))
        except Exception as exc:
            checks.append((label, False, str(exc)))
    try:
        import run
        checks.append(("assessment runner", True, str(Path(run.__file__).resolve())))
    except Exception as exc:
        checks.append(("assessment runner", False, str(exc)))
    try:
        from mcp.server.mcpserver import MCPServer  # noqa: F401
        checks.append(("MCP server API", True, "MCPServer import available"))
    except Exception as exc:
        checks.append(("MCP server API", False, str(exc)))
    for label, ok, detail in checks:
        print(f"{'OK' if ok else 'MISSING'}  {label}: {detail}")
    return 0 if all(x[1] for x in checks) else 2


def mcp(args: argparse.Namespace) -> int:
    from mcp_server import main as server_main
    # Forward parsed options to the existing server entry point. Defining these
    # options here (rather than argparse.REMAINDER) permits --options normally.
    forwarded = ["--workspace", args.workspace, "--max-jobs", str(args.max_jobs),
                 "--assessment-timeout", str(args.assessment_timeout)]
    for path in args.allow_path:
        forwarded.extend(["--allow-path", path])
    sys.argv = ["advassess mcp", *forwarded]
    server_main()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="advassess", description="Agent-driven adversarial ML and AI security assessment")
    parser.add_argument("--version", action="version", version="advassess 0.2.0")
    sub = parser.add_subparsers(dest="command", required=True)
    p_setup = sub.add_parser("setup", help="review and configure MCP for supported agent clients")
    p_setup.add_argument("--clients", help="comma-separated clients: " + ",".join(CLIENTS))
    p_setup.add_argument("--workspace", default=".", help="existing assessment workspace (default: current directory)")
    p_setup.add_argument("--yes", action="store_true", help="apply the displayed plan without an interactive prompt")
    p_setup.set_defaults(func=setup)
    p_un = sub.add_parser("uninstall", help="remove Advassess MCP entries from agent configurations")
    p_un.add_argument("--clients", help="comma-separated client names; omit for interactive selection")
    p_un.add_argument("--yes", action="store_true", help="apply removal without an interactive prompt")
    p_un.set_defaults(func=uninstall)
    p_doc = sub.add_parser("doctor", help="check installation and optional engine dependencies")
    p_doc.set_defaults(func=doctor)
    p_mcp = sub.add_parser("mcp", help="start the MCP server over stdio")
    p_mcp.add_argument("--workspace", default=".", help="assessment workspace (default: current directory)")
    p_mcp.add_argument("--max-jobs", type=int, default=1, help="maximum concurrent assessment jobs")
    p_mcp.add_argument("--allow-path", action="append", default=[], metavar="DIR", help="explicitly allow an external directory; repeatable")
    p_mcp.add_argument("--assessment-timeout", type=int, default=3600, help="hard assessment deadline in seconds")
    p_mcp.set_defaults(func=mcp)
    p_run = sub.add_parser("run", help="run an assessment using an existing YAML config")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("runner_args", nargs=argparse.REMAINDER)
    p_run.set_defaults(func=lambda a: _run_assessment(a))
    p_attacks = sub.add_parser("list-attacks", help="list available attacks")
    p_attacks.set_defaults(func=lambda _: _run_module(["--list-attacks"]))
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (ValueError, OSError) as exc:
        print(f"advassess: error: {exc}", file=sys.stderr)
        return 2


def _run_assessment(args: argparse.Namespace) -> int:
    return _run_module(["--config", args.config, *args.runner_args])


def _run_module(arguments: list[str]) -> int:
    import run
    return int(run.main(arguments) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
