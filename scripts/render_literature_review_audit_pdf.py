"""Render the literature-review update audit Markdown into a compact PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "literature_review_update_audit.md"
OUTPUT = ROOT / "output" / "pdf" / "literature_review_update_audit.pdf"
SCHEMATIC = ROOT / "docs" / "figures" / "literature_repair_gap_schematic.png"


def inline_markdown(text: str) -> str:
    """Handle the small Markdown subset used in the audit source."""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def add_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(doc.leftMargin, 0.35 * inch, "Literature review update audit")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def build_story() -> list:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "AuditTitle",
        parent=styles["Title"],
        fontName="Helvetica",
        fontSize=20,
        leading=24,
        alignment=1,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "AuditSubtitle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        alignment=1,
        spaceAfter=20,
    )
    h2 = ParagraphStyle(
        "AuditH2",
        parent=styles["Heading1"],
        fontName="Helvetica",
        fontSize=16,
        leading=20,
        spaceBefore=12,
        spaceAfter=8,
    )
    h3 = ParagraphStyle(
        "AuditH3",
        parent=styles["Heading2"],
        fontName="Helvetica",
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "AuditBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12.5,
        spaceAfter=7,
    )
    bullet = ParagraphStyle(
        "AuditBullet",
        parent=body,
        leftIndent=14,
        firstLineIndent=0,
        bulletIndent=0,
        spaceAfter=5,
    )
    table_head = ParagraphStyle(
        "AuditTableHead",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10.5,
        spaceBefore=4,
        spaceAfter=5,
    )
    table_item = ParagraphStyle(
        "AuditTableItem",
        parent=body,
        fontSize=8.5,
        leading=10.5,
        leftIndent=12,
        firstLineIndent=0,
        bulletIndent=0,
        spaceAfter=4,
    )

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story: list = [
        Paragraph("Literature Review Update Audit", title),
        Paragraph(
            "Temporal, user-specific, evidence-conflict-aware, counterfactual "
            "minimal-change repair for weather-sensitive multi-day itineraries.",
            subtitle,
        ),
    ]

    if SCHEMATIC.exists():
        story.append(scaled_image(SCHEMATIC, 5.9 * inch, 3.5 * inch))
        story.append(
            Paragraph(
                "Figure 1. Conceptual gap map generated with the available imagegen skill; "
                "exact relationships are documented in the Markdown Mermaid source.",
                body,
            )
        )
        story.append(Spacer(1, 0.12 * inch))

    table_header: list[str] | None = None
    skipped_first_title = False

    for raw in lines:
        line = raw.strip()
        if not line:
            table_header = None
            story.append(Spacer(1, 0.04 * inch))
            continue

        if line.startswith("# "):
            if not skipped_first_title:
                skipped_first_title = True
                continue
            story.append(Paragraph(inline_markdown(line[2:]), h2))
            table_header = None
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markdown(line[3:]), h2))
            table_header = None
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markdown(line[4:]), h3))
            table_header = None
            continue
        if line.startswith(">"):
            story.append(Paragraph(inline_markdown(line.lstrip("> ")), body))
            table_header = None
            continue

        if line.startswith("|") and line.endswith("|"):
            cells = split_table_row(line)
            if is_separator_row(cells):
                continue
            if table_header is None:
                table_header = cells
                story.append(Paragraph(inline_markdown(" / ".join(cells)), table_head))
                continue
            pairs = []
            for header, value in zip(table_header, cells, strict=False):
                if value:
                    pairs.append(f"<b>{inline_markdown(header)}:</b> {inline_markdown(value)}")
            story.append(Paragraph(" | ".join(pairs), table_item, bulletText="-"))
            continue

        table_header = None
        numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if numbered:
            story.append(Paragraph(inline_markdown(numbered.group(2)), bullet, bulletText=numbered.group(1)))
        elif line.startswith("- "):
            story.append(Paragraph(inline_markdown(line[2:]), bullet, bulletText="-"))
        else:
            story.append(Paragraph(inline_markdown(line), body))

    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    doc.build(build_story(), onFirstPage=add_footer, onLaterPages=add_footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
