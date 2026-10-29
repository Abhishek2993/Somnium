"""
config.py — central place to read .env feature flags for the Somnium extension.
"""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))


def _flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ('1', 'true', 'yes', 'on')


ENABLE_SLEEP_MODULE = _flag('ENABLE_SLEEP_MODULE', False)
ENABLE_DISTILBERT = _flag('ENABLE_DISTILBERT', False)
ENABLE_FEATURE_PROFILE = _flag('ENABLE_FEATURE_PROFILE', False)

RANDOM_SEED = int(os.environ.get('SOMNIUM_RANDOM_SEED', '42'))
