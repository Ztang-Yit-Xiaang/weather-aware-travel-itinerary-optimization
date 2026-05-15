"""Write a non-blocking Vulture dead-code report."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "results" / "quality" / "vulture_report.txt"


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "vulture",
        "src/itinerary_system",
        "scripts",
        "--min-confidence",
        "70",
    ]
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report = [
        "$ " + " ".join(command),
        "",
        "stdout:",
        process.stdout.strip() or "<empty>",
        "",
        "stderr:",
        process.stderr.strip() or "<empty>",
        "",
        f"vulture_exit_code: {process.returncode}",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Vulture report written to {REPORT_PATH}")
    if process.returncode not in {0, 3}:
        print("Vulture could not complete cleanly; see report for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
