"""
download_sleep_edf.py — downloads Sleep-EDF Expanded recordings via
mne's PhysioNet helper. NOT run in this dev environment: the full
Sleep-EDF Expanded corpus is several GB and this is a one-time,
user-initiated download, not something to trigger automatically.

Usage (once `mne` is installed -- see backend/requirements.txt):
    backend/venv/bin/python3 sleep/download_sleep_edf.py --subjects 0 1 2
"""
import argparse
import os

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')


def download_subjects(subjects: list[int], recording: list[int] | None = None) -> list[str]:
    """
    Downloads Sleep Cassette recordings for the given subject numbers
    (0-82) via mne.datasets.sleep_physionet.age.fetch_data(). Returns
    the list of downloaded file paths. Requires `mne` and a real
    internet connection; not exercised by the test suite.
    """
    from mne.datasets.sleep_physionet.age import fetch_data

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.environ.setdefault('MNE_DATA', RAW_DATA_DIR)
    paths = fetch_data(subjects=subjects, recording=recording or [1, 2], path=RAW_DATA_DIR)
    return [p for pair in paths for p in pair]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', type=int, nargs='+', default=[0],
                         help='Subject numbers (0-82) to download. Each is ~100-200MB.')
    parser.add_argument('--recording', type=int, nargs='+', default=None, help='Recording numbers (1, 2, or both).')
    args = parser.parse_args()

    print(
        f'Downloading Sleep-EDF data for subjects {args.subjects} into {RAW_DATA_DIR} '
        f'(gitignored -- see sleep/README.md). This can be several hundred MB per subject.'
    )
    paths = download_subjects(args.subjects, args.recording)
    for p in paths:
        print(f'  {p}')


if __name__ == '__main__':
    main()
