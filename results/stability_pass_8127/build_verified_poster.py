"""Build a labeled montage from verified full-size browser screenshots only."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent / "verified_20260810"
OUTPUT = ROOT / "poster_current_webpage_8127.png"
PANELS = (
    ("Initial Trip · 1440×900", "initial_trip_1440x900.png"),
    ("After reload · 1440×900", "after_reload_1440x900.png"),
    ("After Compare → Trip · 1440×900", "after_compare_trip_switch_1440x900.png"),
    ("Compare · Recommended selected · 1024×768", "compare_recommended_1024x768.png"),
    ("Text route panel · 1280×800", "text_route_panel_1280x800.png"),
    ("Mobile Trip · 390×844", "mobile_trip_390x844.png"),
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/seguisb.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    ratio = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    return image.resize(size, Image.Resampling.LANCZOS)


def main() -> None:
    for _, name in PANELS:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(ROOT / name)

    canvas = Image.new("RGB", (1500, 2050), "#eef4f1")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, bold=True)
    label_font = _font(22, bold=True)
    note_font = _font(18)
    draw.text((60, 42), "Itinerary Repair Copilot · Stability Evidence", fill="#122535", font=title_font)
    draw.text(
        (60, 98),
        "Pixel-preserving montage of real browser captures; full-size PNGs remain the acceptance evidence.",
        fill="#526872",
        font=note_font,
    )

    cards = (
        (50, 150, 700, 560),
        (750, 150, 700, 560),
        (50, 740, 700, 560),
        (750, 740, 700, 560),
        (50, 1330, 700, 650),
        (750, 1330, 700, 650),
    )
    for (label, name), (x, y, width, height) in zip(PANELS, cards, strict=True):
        draw.rounded_rectangle(
            (x, y, x + width, y + height), radius=18, fill="#ffffff", outline="#cfdcd7", width=2
        )
        draw.text((x + 20, y + 16), label, fill="#122535", font=label_font)
        with Image.open(ROOT / name) as source:
            rendered = _fit(source.convert("RGB"), width - 40, height - 78)
        px = x + (width - rendered.width) // 2
        py = y + 60 + (height - 70 - rendered.height) // 2
        canvas.paste(rendered, (px, py))

    draw.text(
        (60, 2012),
        "Source directory: results/stability_pass_8127/verified_20260810",
        fill="#526872",
        font=note_font,
    )
    canvas.save(OUTPUT, format="PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
