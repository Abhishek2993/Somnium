"""
dream_features.py — flattens the features/ package's per-text outputs
into a fixed-order numeric feature vector, for models that combine
TF-IDF with hand-engineered features (see train_logreg_tfidf_features.py).
"""
import paths  # noqa: F401  (sets up sys.path for backend/ and features/)

import numpy as np

from emotion import emotion_arc
from hvdc import hvdc_counts
from narrative import narrative_profile
from sensory import sensory_word_ratios

NUMERIC_FEATURE_NAMES = [
    'sensory_visual', 'sensory_auditory', 'sensory_tactile',
    'sensory_olfactory', 'sensory_gustatory', 'sensory_total',
    'hvdc_characters', 'hvdc_aggression', 'hvdc_friendliness',
    'hvdc_misfortune', 'hvdc_good_fortune',
    'hvdc_setting_indoor', 'hvdc_setting_outdoor', 'hvdc_setting_fantastical',
    'narrative_length', 'narrative_first_person_ratio',
    'narrative_tense_past', 'narrative_tense_present',
    'emotion_start', 'emotion_end', 'emotion_slope',
]


def numeric_features_for_text(text: str) -> list[float]:
    sens = sensory_word_ratios(text)
    hv = hvdc_counts(text)
    narr = narrative_profile(text)
    arc = emotion_arc(text)
    return [
        sens['visual'], sens['auditory'], sens['tactile'], sens['olfactory'], sens['gustatory'], sens['total'],
        float(hv['characters']), float(hv['aggression']), float(hv['friendliness']),
        float(hv['misfortune']), float(hv['good_fortune']),
        float(hv['settings']['indoor']), float(hv['settings']['outdoor']), float(hv['settings']['fantastical']),
        float(narr['length']), narr['first_person_ratio'],
        1.0 if narr['tense'] == 'past' else 0.0,
        1.0 if narr['tense'] == 'present' else 0.0,
        arc['start'], arc['end'], arc['slope'],
    ]


def numeric_feature_matrix(texts: list[str]) -> np.ndarray:
    return np.array([numeric_features_for_text(t) for t in texts], dtype=float)
