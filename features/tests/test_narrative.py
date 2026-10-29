from narrative import dominant_tense, first_person_ratio, narrative_profile, word_count


def test_word_count_basic():
    assert word_count("I walked through the forest.") == 5
    assert word_count('') == 0


def test_dominant_tense_unknown_for_empty_text():
    assert dominant_tense('') == 'unknown'


def test_dominant_tense_past_for_past_tense_text():
    assert dominant_tense("I walked through the forest and saw a dragon.") == 'past'


def test_dominant_tense_present_for_present_tense_text():
    assert dominant_tense("I am walking through the forest and I see a dragon.") == 'present'


def test_first_person_ratio_zero_for_empty_text():
    assert first_person_ratio('') == 0.0


def test_first_person_ratio_counts_first_person_pronouns():
    ratio = first_person_ratio("I walked with my friend and we saw a dragon")
    assert ratio == 3 / 10


def test_first_person_ratio_zero_when_no_first_person_pronouns():
    assert first_person_ratio("He walked with his friend and they saw a dragon") == 0.0


def test_narrative_profile_shape():
    profile = narrative_profile("I walked through the forest and saw a dragon.")
    assert set(profile.keys()) == {'length', 'tense', 'first_person_ratio'}
    assert profile['length'] == 9
    assert profile['tense'] == 'past'
