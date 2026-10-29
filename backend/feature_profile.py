"""
feature_profile.py — builds the optional /api/analyze feature profile
(emotion arc, bizarreness, sensory ratios, lucid probability) from the
features/ package. Only imported when ENABLE_FEATURE_PROFILE is on
(see app.py), so the sentence-embedding model bizarreness.py needs
isn't loaded at all when the flag is off.
"""
import os
import sys

_FEATURES_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'features'))
if _FEATURES_DIR not in sys.path:
    sys.path.insert(0, _FEATURES_DIR)


def build_feature_profile(text: str) -> dict:
    from bizarreness import bizarreness_score
    from emotion import emotion_arc
    from lucidity import lucid_probability
    from sensory import sensory_word_ratios

    return {
        'emotion_arc': emotion_arc(text),
        'bizarreness': bizarreness_score(text),
        'sensory_ratios': sensory_word_ratios(text),
        'lucid_probability': lucid_probability(text),
    }
