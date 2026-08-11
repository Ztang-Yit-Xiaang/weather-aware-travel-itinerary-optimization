from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "itinerary_system" / "product_app" / "static"
CSS = STATIC / "css" / "app.css"
INDEX = STATIC / "index.html"


def _css() -> str:
    return CSS.read_text(encoding="utf-8")


def test_critical_controls_and_disclosures_have_44px_targets() -> None:
    css = _css()

    assert (
        ".icon-button { width: 44px; min-width: 44px; height: 44px; "
        "min-height: 44px;" in css
    )
    assert ".toolbar-button { min-width: 44px; min-height: 44px;" in css
    assert ".map-mode-switch button { min-width: 72px; min-height: 44px;" in css
    assert ".alternative-card button { width: 100%; min-height: 44px;" in css
    assert ".download-link { display: inline-flex; align-items: center; min-height: 44px;" in css
    assert (
        ".textual-map-alternative summary, .evidence-card summary { "
        "min-height: 44px; display: flex; align-items: center;" in css
    )
    assert ".mobile-nav button { min-width: 0; min-height: 44px;" in css


def test_mobile_map_uses_compact_scrollable_bottom_sheet_treatment() -> None:
    css = _css()
    mobile = css[css.index("@media (max-width: 430px)") :]

    assert "#geographic-map.geographic-map { inset: 0; }" in mobile
    assert "bottom: 112px" in mobile
    assert "max-height: min(32dvh, 230px)" in mobile
    assert "max-height: 104px" in mobile
    assert "overflow-y: auto" in mobile
    assert ".map-legend span { flex: 0 0 auto; }" in mobile


def test_mobile_toolbar_and_evidence_content_do_not_silently_clip() -> None:
    css = _css()

    assert "max-width: 100vw" in css
    assert "scroll-snap-type: inline proximity" in css
    assert "scrollbar-width: thin" in css
    assert ".comparison-table { width: 100%; table-layout: fixed;" in css
    assert "overflow-wrap: anywhere; word-break: break-word" in css
    assert ".evidence-card dd { padding-bottom: 7px; word-break: break-word; }" in css
    assert "html, body { width: 100%; max-width: 100%;" in css


def test_about_remains_available_and_route_swaps_are_not_one_large_live_region() -> None:
    markup = INDEX.read_text(encoding="utf-8")

    assert 'id="help-button" type="button" aria-label="About this demo"' in markup
    assert '<section id="workspace-content" class="workspace-content" aria-label="Workspace view"></section>' in markup
    assert 'id="workspace-content" class="workspace-content" aria-live=' not in markup
    assert ".top-actions .icon-button { display: none; }" not in _css()


def test_recovery_and_support_text_uses_readable_sizes() -> None:
    css = _css()

    assert ".runtime-banner" in css and "font-size: 13px" in css
    assert ".map-context-card" in css and "font-size: 13px" in css
    assert ".textual-map-alternative" in css and "font-size: 13px" in css
    assert ".evidence-card dl" in css and "font-size: 12px" in css
    assert ".truth-note" in css and "font-size: 12px" in css
