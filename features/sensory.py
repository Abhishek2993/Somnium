"""
sensory.py — ratio of sensory words (by modality) to total words in a
dream text. Uses a small hand-curated lexicon per sense (not an
exhaustive or externally-validated list -- see SENSORY_LEXICON below;
TODO: swap in a published sensory-word list if one is sourced later).
"""
import re

SENSORY_LEXICON = {
    'visual': {
        'see', 'saw', 'seen', 'look', 'looked', 'looking', 'watch', 'watched',
        'bright', 'dark', 'glow', 'glowing', 'glowed', 'shine', 'shining', 'shone',
        'color', 'colour', 'colorful', 'colourful', 'light', 'lit', 'shadow', 'shadowy',
        'red', 'blue', 'green', 'yellow', 'golden', 'white', 'black', 'blurry', 'vivid',
        'sparkle', 'sparkling', 'flash', 'flashing', 'gleam', 'gleaming',
    },
    'auditory': {
        'hear', 'heard', 'hearing', 'sound', 'sounds', 'sounded', 'noise', 'noisy',
        'loud', 'quiet', 'silent', 'silence', 'whisper', 'whispered', 'whispering',
        'scream', 'screamed', 'screaming', 'shout', 'shouted', 'shouting', 'music',
        'sing', 'singing', 'sang', 'echo', 'echoed', 'echoing', 'ring', 'ringing', 'roar', 'roaring',
    },
    'tactile': {
        'touch', 'touched', 'touching', 'feel', 'felt', 'feeling', 'rough', 'smooth',
        'soft', 'hard', 'cold', 'warm', 'hot', 'wet', 'dry', 'sticky', 'sharp',
        'grip', 'gripped', 'gripping', 'squeeze', 'squeezed', 'texture', 'pressure',
        'tingle', 'tingling', 'numb', 'shiver', 'shivering',
    },
    'olfactory': {
        'smell', 'smelled', 'smelt', 'smelling', 'scent', 'odor', 'odour', 'fragrance',
        'stink', 'stinking', 'stench', 'perfume', 'aroma', 'musty', 'fresh',
    },
    'gustatory': {
        'taste', 'tasted', 'tasting', 'sweet', 'sour', 'bitter', 'salty', 'spicy',
        'delicious', 'flavor', 'flavour', 'bland', 'savory', 'savoury',
    },
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or '')]


def sensory_word_ratios(text: str) -> dict:
    """
    Ratio of sensory words (per modality, and combined) to total word
    count. Returns {'visual': r, 'auditory': r, ..., 'total': r,
    'word_count': n}, all ratios 0.0 if the text has no words.
    """
    tokens = tokenize(text)
    n = len(tokens)
    if n == 0:
        return {**{sense: 0.0 for sense in SENSORY_LEXICON}, 'total': 0.0, 'word_count': 0}

    counts = {sense: sum(1 for t in tokens if t in words) for sense, words in SENSORY_LEXICON.items()}
    ratios = {sense: count / n for sense, count in counts.items()}
    ratios['total'] = sum(counts.values()) / n
    ratios['word_count'] = n
    return ratios
