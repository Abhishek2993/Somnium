"""
dreambank_loader.py — optional loader for a local DreamBank-style text file
(https://dreambank.net), stored in the dreambank_dreams table.

Usage:
    # in backend/.env: DREAMBANK_FILE_PATH=/path/to/series.txt
    backend/venv/bin/python3 data/dreambank_loader.py
    # or explicitly:
    backend/venv/bin/python3 data/dreambank_loader.py --file /path/to/series.txt

Format assumption
------------------
DreamBank series are typically exported as plain text with one dream per
paragraph, separated by a blank line, sometimes preceded by a numeric/date
header line (e.g. "1.", "#42", "3/14/1985"). This loader splits on blank
lines and strips a short numeric/date-only header line if present.

TODO: this has not been run against a real DreamBank export in this
environment (no DreamBank file was available to test against) -- verify
the paragraph-splitting assumption against an actual exported file before
relying on its output, and adjust split_into_dreams()/strip_header() if
the real format differs (e.g. HTML export instead of plain text).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend', '.env'))

import db  # noqa: E402
from text_cleaning import content_hash, passes_filters, word_count  # noqa: E402

_HEADER_LINE_RE = re.compile(r'^\s*[#]?\s*[\d./-]{1,12}\.?\s*$')


def split_into_dreams(raw_text: str) -> list[str]:
    """Split a DreamBank text export into individual dream paragraphs."""
    if not raw_text:
        return []
    paragraphs = re.split(r'\n\s*\n+', raw_text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def strip_header(paragraph: str) -> str:
    """Remove a leading numeric/date-only header line, if present."""
    lines = paragraph.split('\n')
    if len(lines) > 1 and _HEADER_LINE_RE.match(lines[0]):
        return '\n'.join(lines[1:]).strip()
    return paragraph.strip()


def load_dreambank_file(path: str, min_words: int = 50) -> list[dict]:
    """Parse a DreamBank text file into dreambank_dreams-shaped records."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        raw_text = f.read()

    source_label = os.path.basename(path)
    records = []
    for paragraph in split_into_dreams(raw_text):
        text = strip_header(paragraph)
        if not passes_filters(text, min_words=min_words):
            continue
        records.append({
            'source_label': source_label,
            'text': text,
            'word_count': word_count(text),
            'content_hash': content_hash(text),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', default=os.environ.get('DREAMBANK_FILE_PATH'))
    parser.add_argument('--min-words', type=int, default=50)
    args = parser.parse_args()

    if not args.file:
        raise SystemExit(
            'No DreamBank file given. Pass --file or set DREAMBANK_FILE_PATH in backend/.env.'
        )

    db.init_db()
    records = load_dreambank_file(args.file, min_words=args.min_words)
    stored = 0
    for record in records:
        if db.insert_dreambank_dream(record) is not None:
            stored += 1
    print(f'Parsed {len(records)} dreams from {args.file}, stored {stored} new rows.')


if __name__ == '__main__':
    main()
