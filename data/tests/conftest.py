import os
import sys

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(DATA_DIR, '..', 'backend')
for path in (DATA_DIR, os.path.abspath(BACKEND_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
