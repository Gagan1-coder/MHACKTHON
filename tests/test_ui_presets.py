"""
Unit tests confirming that the preset dropdown selector has been completely removed
from the Context Surgeon frontend (HTML and JavaScript).
"""

from pathlib import Path


def test_preset_dropdown_completely_removed():
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    index_html_path = frontend_dir / "index.html"
    assert index_html_path.exists(), "frontend/index.html not found"

    html_content = index_html_path.read_text(encoding="utf-8")

    # Verify preset dropdown elements and labels are removed
    assert "example-presets" not in html_content, "id='example-presets' still found in index.html"
    assert "preset-selector-group" not in html_content, "preset-selector-group class still found in index.html"
    assert "Load Test Preset:" not in html_content, "'Load Test Preset:' label still found in index.html"
    assert "-- Choose a Preset Scenario --" not in html_content, "preset default option still found in index.html"


def test_app_js_preset_selector_removed():
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    app_js_path = frontend_dir / "app.js"
    assert app_js_path.exists(), "frontend/app.js not found"

    app_js_content = app_js_path.read_text(encoding="utf-8")

    # Verify no presetSelect element reference or PRESETS dictionary
    assert "const PRESETS" not in app_js_content, "PRESETS dictionary still found in app.js"
    assert "example-presets" not in app_js_content, "'example-presets' ID still referenced in app.js"
    assert "presetSelect" not in app_js_content, "presetSelect still referenced in app.js"
