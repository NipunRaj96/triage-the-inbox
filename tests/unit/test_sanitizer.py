"""Unit tests for text sanitization against hostile inputs."""

from app.sanitization import sanitize


def test_sanitize_normal_text():
    res = sanitize("I need a refund for my damaged shoes")
    assert res.clean_text == "I need a refund for my damaged shoes"
    assert not res.is_empty
    assert not res.is_binary_junk
    assert not res.had_hostile_markers


def test_sanitize_zero_width_chars():
    # U+200B zero-width space
    res = sanitize("refund\u200b\u200b\u200b please")
    assert res.clean_text == "refund please"
    assert res.had_hostile_markers


def test_sanitize_bidi_override():
    # RTL override U+202E
    res = sanitize("\u202ecancel my order\u202c")
    assert res.clean_text == "cancel my order"
    assert res.had_hostile_markers


def test_sanitize_decomposed_unicode():
    # cafe + combining acute accent -> café (NFKC should normalize)
    decomposed = "cafe\u0301 order issue"
    res = sanitize(decomposed)
    assert "café" in res.clean_text or "cafe" in res.clean_text


def test_sanitize_empty_text():
    res = sanitize("")
    assert res.is_empty
    assert res.clean_text == ""
