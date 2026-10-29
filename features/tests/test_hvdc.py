from hvdc import hvdc_counts


def test_hvdc_counts_empty_text():
    result = hvdc_counts('')
    assert result['word_count'] == 0
    assert result['characters'] == 0
    assert result['settings'] == {'indoor': 0, 'outdoor': 0, 'fantastical': 0}


def test_hvdc_counts_detects_characters():
    result = hvdc_counts("My mother and my friend were there, and a stranger watched us")
    assert result['characters'] >= 3


def test_hvdc_counts_detects_aggression_and_friendliness():
    result = hvdc_counts("He attacked me and I fought back, but then we hugged and laughed")
    assert result['aggression'] >= 2
    assert result['friendliness'] >= 2


def test_hvdc_counts_detects_misfortune_and_good_fortune():
    result = hvdc_counts("I fell and got trapped, but then I escaped and found my way home")
    assert result['misfortune'] >= 2
    assert result['good_fortune'] >= 1


def test_hvdc_counts_detects_settings():
    result = hvdc_counts("I was in the kitchen, then outside near the forest, then in a castle")
    assert result['settings']['indoor'] >= 1
    assert result['settings']['outdoor'] >= 1
    assert result['settings']['fantastical'] >= 1
