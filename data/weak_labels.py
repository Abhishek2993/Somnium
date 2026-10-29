"""
weak_labels.py — derive weak (noisy, not gold) labels for a Reddit dream
post from its subreddit and flair. These are NOT used for reported
metrics (see data/README.md and data/gold_template.csv) -- they're a
cheap signal for the lucid/nightmare classifiers in features/lucidity.py
and features/nightmare.py to bootstrap from before gold labels exist.
"""
import re

LUCID_SUBREDDITS = {'luciddreaming'}
NIGHTMARE_SUBREDDITS = {'nightmares'}

LUCID_FLAIR_RE = re.compile(r'\blucid\b', re.IGNORECASE)
NIGHTMARE_FLAIR_RE = re.compile(r'\b(nightmare|night\s?terror)\b', re.IGNORECASE)


def weak_is_lucid(subreddit: str, flair: str | None) -> bool:
    """True if the subreddit or flair marks this post as a lucid dream."""
    if subreddit.lower() in LUCID_SUBREDDITS:
        return True
    return bool(flair and LUCID_FLAIR_RE.search(flair))


def weak_is_nightmare(subreddit: str, flair: str | None) -> bool:
    """True if the subreddit or flair marks this post as a nightmare."""
    if subreddit.lower() in NIGHTMARE_SUBREDDITS:
        return True
    return bool(flair and NIGHTMARE_FLAIR_RE.search(flair))


def weak_label(subreddit: str, flair: str | None) -> dict:
    """Combined weak-label dict for a reddit_dreams row."""
    return {
        'weak_lucid': weak_is_lucid(subreddit, flair),
        'weak_nightmare': weak_is_nightmare(subreddit, flair),
    }
