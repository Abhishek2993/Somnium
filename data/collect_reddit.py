"""
collect_reddit.py — collect dream posts from r/dreams, r/LucidDreaming and
r/Nightmares via PRAW, clean + filter them, and store them in the
reddit_dreams table.

Usage:
    cp backend/.env.example backend/.env   # fill in REDDIT_CLIENT_ID etc.
    backend/venv/bin/python3 data/collect_reddit.py
    backend/venv/bin/python3 data/collect_reddit.py --subreddits dreams --limit 50

Design notes
------------
- Never stores usernames: submission.author is read only to detect a
  deleted/removed post, never persisted.
- Resumable: each subreddit's post ids are appended to
  data/post_ids/<subreddit>.txt as they're processed (whether or not they
  passed the filters). A re-run skips any id already in that file, so an
  interrupted run picks up where it left off without re-hitting PRAW for
  posts it already looked at. The DB layer also enforces this: post_id and
  content_hash are UNIQUE columns, so even without the checkpoint file a
  re-run would not create duplicate rows.
- Rate limits: PRAW/prawcore sleeps automatically based on the
  X-Ratelimit-* response headers; REQUEST_DELAY_SECONDS below adds a small
  extra pause between submissions as an additional courtesy.
- Raw collected text is never committed to git (see data/README.md and the
  root .gitignore) -- only this script and the post-id lists are.
"""
import argparse
import os
import sys
import time

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend')
sys.path.insert(0, os.path.abspath(BACKEND_DIR))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(BACKEND_DIR, '.env'))

import db  # noqa: E402
from text_cleaning import clean_body, content_hash, passes_filters, word_count  # noqa: E402

SUBREDDITS = ['dreams', 'LucidDreaming', 'Nightmares']
REQUEST_DELAY_SECONDS = 1.0
POST_IDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'post_ids')


def _checkpoint_path(subreddit: str) -> str:
    return os.path.join(POST_IDS_DIR, f'{subreddit.lower()}.txt')


def load_checkpoint(subreddit: str) -> set:
    path = _checkpoint_path(subreddit)
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def append_checkpoint(subreddit: str, post_id: str) -> None:
    os.makedirs(POST_IDS_DIR, exist_ok=True)
    with open(_checkpoint_path(subreddit), 'a', encoding='utf-8') as f:
        f.write(post_id + '\n')
        f.flush()
        os.fsync(f.fileno())


def build_record(submission) -> dict:
    """
    Turn a PRAW Submission (or any duck-typed object with the same
    attributes) into a reddit_dreams row dict. Pure given its input --
    does not touch the network, the DB, or the checkpoint file.
    """
    raw_body = getattr(submission, 'selftext', '') or ''
    body = clean_body(raw_body)
    return {
        'post_id': submission.id,
        'subreddit': str(submission.subreddit),
        'created_utc': float(submission.created_utc),
        'flair': getattr(submission, 'link_flair_text', None),
        'title': submission.title,
        'body': body,
        'score': int(submission.score),
        'num_comments': int(submission.num_comments),
        'word_count': word_count(body),
        'content_hash': content_hash(submission.title + ' ' + body),
    }


def is_deleted_account(submission) -> bool:
    """True if the post's author account is gone (a sign of a removed post)."""
    return getattr(submission, 'author', None) is None


def should_store(submission, record: dict, min_words: int = 50) -> bool:
    if is_deleted_account(submission):
        return False
    return passes_filters(record['body'], min_words=min_words)


def make_reddit_client():
    import praw
    client_id = os.environ.get('REDDIT_CLIENT_ID')
    client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
    user_agent = os.environ.get('REDDIT_USER_AGENT')
    if not (client_id and client_secret and user_agent):
        raise RuntimeError(
            'Missing Reddit credentials. Copy backend/.env.example to backend/.env '
            'and fill in REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT.'
        )
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


def collect_subreddit(reddit, subreddit_name: str, limit: int, min_words: int) -> dict:
    """Collect up to `limit` new/top posts from one subreddit. Returns counters."""
    done_ids = load_checkpoint(subreddit_name)
    stats = {'seen': 0, 'stored': 0, 'skipped_checkpoint': 0, 'filtered_out': 0}

    subreddit = reddit.subreddit(subreddit_name)
    listings = [subreddit.new(limit=limit), subreddit.top(time_filter='all', limit=limit)]

    for listing in listings:
        for submission in listing:
            stats['seen'] += 1
            if submission.id in done_ids:
                stats['skipped_checkpoint'] += 1
                continue

            record = build_record(submission)
            if should_store(submission, record, min_words=min_words):
                row_id = db.insert_reddit_dream(record)
                if row_id is not None:
                    stats['stored'] += 1
            else:
                stats['filtered_out'] += 1

            done_ids.add(submission.id)
            append_checkpoint(subreddit_name, submission.id)
            time.sleep(REQUEST_DELAY_SECONDS)

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subreddits', nargs='+', default=SUBREDDITS)
    parser.add_argument('--limit', type=int, default=1000, help='max posts per listing per subreddit')
    parser.add_argument('--min-words', type=int, default=50)
    args = parser.parse_args()

    db.init_db()
    reddit = make_reddit_client()

    for name in args.subreddits:
        print(f'Collecting r/{name} ...')
        stats = collect_subreddit(reddit, name, limit=args.limit, min_words=args.min_words)
        print(f'  r/{name}: {stats}')


if __name__ == '__main__':
    main()
