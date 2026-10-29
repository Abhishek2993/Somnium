"""
paths.py — sys.path setup shared by every analysis/ script.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _subdir in ('backend', 'features', 'models'):
    _path = os.path.join(_ROOT, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

REPORTS_DIR = os.path.join(_ROOT, 'reports')
