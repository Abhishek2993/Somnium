from datetime import datetime, timezone

from posting_time import posting_hour_histogram, posting_hour_histogram_by_subreddit


def _ts(hour, subreddit='dreams'):
    return {
        'created_utc': datetime(2026, 1, 1, hour, 0, 0, tzinfo=timezone.utc).timestamp(),
        'subreddit': subreddit,
    }


def test_posting_hour_histogram_covers_all_24_hours():
    histogram = posting_hour_histogram([_ts(3), _ts(3), _ts(14)])
    assert set(histogram.keys()) == set(range(24))
    assert histogram[3] == 2
    assert histogram[14] == 1
    assert histogram[0] == 0


def test_posting_hour_histogram_empty_rows():
    histogram = posting_hour_histogram([])
    assert sum(histogram.values()) == 0
    assert len(histogram) == 24


def test_posting_hour_histogram_by_subreddit_groups_correctly():
    rows = [_ts(3, 'dreams'), _ts(3, 'Nightmares'), _ts(5, 'dreams')]
    by_subreddit = posting_hour_histogram_by_subreddit(rows)
    assert set(by_subreddit.keys()) == {'dreams', 'Nightmares'}
    assert by_subreddit['dreams'][3] == 1
    assert by_subreddit['dreams'][5] == 1
    assert by_subreddit['Nightmares'][3] == 1
