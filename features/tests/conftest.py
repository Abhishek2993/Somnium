import os
import sys

FEATURES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FEATURES_DIR not in sys.path:
    sys.path.insert(0, FEATURES_DIR)
