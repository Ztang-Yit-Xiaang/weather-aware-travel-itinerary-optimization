"""Serve the modular dashboard over localhost.

The full dashboard loads JSON/GeoJSON assets with fetch(), so opening it with
file:// can be blocked by browser security rules. Use this helper instead.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD_DIR = REPO_ROOT / "results" / "figures" / "full_interactive_dashboard"


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the full interactive dashboard over localhost.")
    parser.add_argument(
        "--dashboard-dir",
        default=str(DEFAULT_DASHBOARD_DIR),
        help="Dashboard directory to serve. Defaults to results/figures/full_interactive_dashboard/.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Starting port. Defaults to 8000.")
    parser.add_argument("--host", default="localhost", help="Host to bind. Defaults to localhost.")
    parser.add_argument("--open", action="store_true", help="Open the dashboard URL in the default browser.")
    return parser.parse_args()


def make_server(directory: Path, host: str, start_port: int) -> tuple[ReusableTCPServer, str]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    for port in range(int(start_port), int(start_port) + 100):
        try:
            server = ReusableTCPServer((host, port), handler)
            return server, f"http://{host}:{port}/"
        except OSError:
            continue
    raise RuntimeError(f"No available port found from {start_port} to {start_port + 99}")


def main() -> int:
    args = parse_args()
    dashboard_dir = Path(args.dashboard_dir).expanduser().resolve()
    if not dashboard_dir.exists():
        print(f"Dashboard directory does not exist: {dashboard_dir}", file=sys.stderr)
        return 1
    if not (dashboard_dir / "index.html").exists():
        print(f"Dashboard index.html does not exist: {dashboard_dir / 'index.html'}", file=sys.stderr)
        return 1

    try:
        server, url = make_server(dashboard_dir, args.host, args.port)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Serving dashboard at:")
    print(url)
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
