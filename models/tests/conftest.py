import os
import sys

MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)
