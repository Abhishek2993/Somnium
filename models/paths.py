"""
paths.py — sys.path setup shared by every models/ script, so they can
`import ml`/`db` (backend/) and `import emotion`/`sensory`/etc. (features/)
as flat modules, same convention as data/ and features/ use.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _subdir in ('backend', 'features'):
    _path = os.path.join(_ROOT, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

REPORTS_DIR = os.path.join(_ROOT, 'reports')
