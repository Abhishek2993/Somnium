"""
text_cleaning.py — pure text-cleaning/filtering helpers shared by
collect_reddit.py and dreambank_loader.py. No network, no DB, no PRAW.
"""
import hashlib
import re

from langdetect import detect, LangDetectException

REMOVED_MARKERS = {'[deleted]', '[removed]', ''}

_URL_RE = re.compile(r'https?://\S+')
_MARKDOWN_LINK_RE = re.compile(r'\[([^\]]*)\]\(([^)]*)\)')
_WHITESPACE_RE = re.compile(r'\s+')


def clean_body(raw: str) -> str:
    """Strip URLs/markdown links and collapse whitespace in a raw post body."""
    if not raw:
        return ''
    text = _MARKDOWN_LINK_RE.sub(r'\1', raw)
    text = _URL_RE.sub('', text)
    text = text.replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<')
    text = _WHITESPACE_RE.sub(' ', text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def is_removed_or_deleted(body: str) -> bool:
    return body.strip() in REMOVED_MARKERS


def is_english(text: str, min_chars: int = 20) -> bool:
    """
    Best-effort English detection. Very short strings are unreliable for
    langdetect, so anything under min_chars is treated as non-English
    (callers should already be filtering on word count >= 50 anyway).
    """
    if len(text.strip()) < min_chars:
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


def content_hash(text: str) -> str:
    """Stable hash used as a cross-post duplicate key (title+body text, not an id)."""
    normalized = _WHITESPACE_RE.sub(' ', text).strip().lower()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def passes_filters(body: str, min_words: int = 50) -> bool:
    if is_removed_or_deleted(body):
        return False
    if word_count(body) < min_words:
        return False
    if not is_english(body):
        return False
    return True
