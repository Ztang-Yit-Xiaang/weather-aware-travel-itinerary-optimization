"""Render human-reviewable exact file and LLM input/output evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid_json_document")
    return value


def fence(value: str) -> str:
    longest = 0
    current = 0
    for character in value:
        if character == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return "`" * max(3, longest + 1)


def body_block(label: str, value: str) -> list[str]:
    marker = fence(value)
    return [f"**{label}**", "", f"{marker}json", value, marker, ""]


def render(output_dir: Path) -> str:
    output_dir = output_dir.resolve()
    file_manifest = read_json(output_dir / "file_io_manifest.json")
    sections = [
        "# G4 Exact File and LLM Input/Output Review",
        "",
        "This report adds exact review evidence without removing any prior aggregate, "
        "case, safety, smoke, or architecture evidence.",
        "",
        "- Scope: synthetic fixed-24 Copilot evaluation only.",
        "- Exact LLM request and response bodies are reproduced byte-for-byte from the transport witness.",
        "- Authorization headers, cookies, API keys, and raw exception messages are intentionally excluded.",
        "- The source JSON files remain authoritative for hashes and machine verification.",
        "",
        "## Exact file inputs",
        "",
        "| Path | Bytes | SHA-256 | Exact content |",
        "|---|---:|---|---|",
    ]
    for row in file_manifest["input_files"]:
        sections.append(
            f"| `{row['path']}` | {row['size_bytes']} | `{row['sha256']}` | `{row['exact_content_location']}` |"
        )
    sections.extend(
        [
            "",
            "## Exact file outputs",
            "",
            "| Path | Bytes | SHA-256 | Exact content |",
            "|---|---:|---|---|",
        ]
    )
    for row in file_manifest["output_files"]:
        sections.append(
            f"| `{row['path']}` | {row['size_bytes']} | `{row['sha256']}` | `{row['exact_content_location']}` |"
        )
    for effort in ("low", "medium"):
        document = read_json(output_dir / f"{effort}_exact_llm_io.json")
        sections.extend(
            [
                "",
                f"## Exact LLM I/O ? {effort}",
                "",
                f"- Model: `{document['model']}`",
                f"- Prompt SHA-256: `{document['prompt_sha256']}`",
                f"- Request-controls SHA-256: `{document['request_controls_sha256']}`",
                f"- Source-snapshot SHA-256: `{document['source_snapshot_sha256']}`",
                f"- Attempts: {len(document['records'])}",
                "",
            ]
        )
        for record in document["records"]:
            sections.extend(
                [
                    f"### {effort} ? {record['case_id']} ? attempt {record['case_attempt']}",
                    "",
                    f"- Sequence: {record['sequence']}",
                    f"- Started: `{record['started_at']}`",
                    f"- Completed: `{record['completed_at']}`",
                    f"- HTTP status: {record['response_status_code']}",
                    f"- Request SHA-256: `{record['request_body_sha256']}`",
                    f"- Response SHA-256: `{record['response_body_sha256']}`",
                    "",
                ]
            )
            sections.extend(body_block("Exact LLM input", record["request_body_utf8"]))
            sections.extend(body_block("Exact LLM output", record["response_body_utf8"]))
    return "\n".join(sections).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = render(args.evidence_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "characters": len(rendered)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
