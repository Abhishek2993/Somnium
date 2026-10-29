"""
hvdc.py — a simplified, word-list-based approximation of
Hall-Van de Castle (HVdC) dream-content coding: character mentions,
aggression/friendliness, misfortune/good-fortune, and setting type.

IMPORTANT: the real HVdC system (Hall & Van de Castle, 1966) is a
detailed manual coded by trained human raters against the full text
and context of a dream report. This module is a lightweight word-list
proxy for exploratory analysis, not a validated HVdC coding -- do not
report its output as equivalent to published HVdC norms.
"""
import re

CHARACTER_WORDS = {
    'he', 'she', 'they', 'him', 'her', 'them', 'his', 'their',
    'man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'children',
    'friend', 'friends', 'stranger', 'strangers', 'mother', 'father',
    'sister', 'brother', 'teacher', 'monster', 'creature', 'animal',
    'dog', 'cat', 'crowd', 'people', 'someone', 'everyone',
}

AGGRESSION_WORDS = {
    'attack', 'attacked', 'attacking', 'hit', 'hits', 'fight', 'fought',
    'fighting', 'chase', 'chased', 'chasing', 'kill', 'killed', 'killing',
    'hurt', 'punch', 'punched', 'stab', 'stabbed', 'scream', 'screamed',
    'threaten', 'threatened', 'yell', 'yelled', 'shove', 'shoved',
}

FRIENDLINESS_WORDS = {
    'help', 'helped', 'helping', 'hug', 'hugged', 'hugging', 'smile',
    'smiled', 'smiling', 'laugh', 'laughed', 'laughing', 'kind', 'kindness',
    'comfort', 'comforted', 'love', 'loved', 'embrace', 'embraced',
    'share', 'shared', 'sharing', 'welcome', 'welcomed',
}

MISFORTUNE_WORDS = {
    'lost', 'lose', 'losing', 'broke', 'broken', 'fell', 'fall', 'falling',
    'died', 'death', 'dying', 'trapped', 'stuck', 'injured', 'injury',
    'hurt', 'failed', 'failure', 'drowned', 'drowning', 'collapse', 'collapsed',
}

GOOD_FORTUNE_WORDS = {
    'found', 'win', 'won', 'winning', 'saved', 'rescue', 'rescued',
    'succeed', 'succeeded', 'success', 'achieve', 'achieved', 'reward',
    'rewarded', 'escaped',
}

SETTING_WORDS = {
    'indoor': {
        'house', 'room', 'kitchen', 'bedroom', 'hallway', 'office', 'school',
        'classroom', 'building', 'apartment', 'basement', 'attic',
    },
    'outdoor': {
        'forest', 'beach', 'street', 'mountain', 'river', 'ocean', 'field',
        'park', 'sky', 'desert', 'jungle', 'garden', 'road',
    },
    'fantastical': {
        'castle', 'dungeon', 'spaceship', 'portal', 'dimension', 'void',
        'underworld', 'kingdom', 'realm', 'dreamscape', 'otherworld',
    },
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or '')]


def _count(tokens: list[str], lexicon: set) -> int:
    return sum(1 for t in tokens if t in lexicon)


def hvdc_counts(text: str) -> dict:
    """
    Simplified HVdC-style counts for a dream text: raw counts (not
    ratios, matching how HVdC indices are traditionally reported per
    report) for characters/aggression/friendliness/misfortune/
    good_fortune, plus a settings sub-dict, and the word count used.
    """
    tokens = _tokenize(text)
    settings = {category: _count(tokens, words) for category, words in SETTING_WORDS.items()}
    return {
        'characters': _count(tokens, CHARACTER_WORDS),
        'aggression': _count(tokens, AGGRESSION_WORDS),
        'friendliness': _count(tokens, FRIENDLINESS_WORDS),
        'misfortune': _count(tokens, MISFORTUNE_WORDS),
        'good_fortune': _count(tokens, GOOD_FORTUNE_WORDS),
        'settings': settings,
        'word_count': len(tokens),
    }
