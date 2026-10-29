from text_cleaning import (
    clean_body, content_hash, is_english, is_removed_or_deleted,
    passes_filters, word_count,
)

LONG_ENGLISH_DREAM = (
    "I was walking through a long hallway that kept stretching further away "
    "every time I took a step forward, and the lights above me flickered in "
    "a strange rhythm while I tried to find a door that would let me out of "
    "this endless corridor before something behind me caught up."
)


def test_clean_body_strips_urls_and_markdown_links():
    raw = "Check [this out](https://example.com/x) and also https://foo.bar/baz directly."
    cleaned = clean_body(raw)
    assert 'http' not in cleaned
    assert 'this out' in cleaned


def test_clean_body_collapses_whitespace():
    assert clean_body("a   b\n\n c") == "a b c"


def test_clean_body_empty_input():
    assert clean_body('') == ''
    assert clean_body(None) == ''


def test_word_count_basic():
    assert word_count("one two three") == 3
    assert word_count('') == 0


def test_is_removed_or_deleted():
    assert is_removed_or_deleted('[deleted]') is True
    assert is_removed_or_deleted('[removed]') is True
    assert is_removed_or_deleted('') is True
    assert is_removed_or_deleted('a real dream') is False


def test_is_english_true_for_long_english_text():
    assert is_english(LONG_ENGLISH_DREAM) is True


def test_is_english_false_for_short_text():
    assert is_english('ok') is False


def test_is_english_false_for_non_english_text():
    assert is_english(
        "Caminaba por un largo pasillo que se extendía cada vez más lejos "
        "mientras las luces parpadeaban sobre mi cabeza de una manera extraña."
    ) is False


def test_content_hash_is_stable_and_case_insensitive():
    a = content_hash("Hello   World")
    b = content_hash("hello world")
    assert a == b
    assert content_hash("Hello World 2") != a


def test_passes_filters_rejects_short_text():
    assert passes_filters("too short", min_words=50) is False


def test_passes_filters_rejects_removed_marker():
    assert passes_filters('[removed]', min_words=1) is False


def test_passes_filters_accepts_long_english_text():
    long_text = (LONG_ENGLISH_DREAM + ' ') * 2
    assert passes_filters(long_text, min_words=10) is True
