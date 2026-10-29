from sensory import sensory_word_ratios, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("I SAW a Bright, Glowing light!") == ['i', 'saw', 'a', 'bright', 'glowing', 'light']


def test_sensory_word_ratios_empty_text():
    result = sensory_word_ratios('')
    assert result['word_count'] == 0
    assert result['total'] == 0.0
    assert result['visual'] == 0.0


def test_sensory_word_ratios_visual_words_detected():
    result = sensory_word_ratios("I saw a bright glowing light shining in the dark")
    assert result['visual'] > 0
    assert result['word_count'] == 10


def test_sensory_word_ratios_mixed_modalities():
    text = "I heard a loud noise, touched something cold and smelled smoke"
    result = sensory_word_ratios(text)
    assert result['auditory'] > 0
    assert result['tactile'] > 0
    assert result['olfactory'] > 0


def test_sensory_word_ratios_total_is_sum_of_modalities():
    text = "I saw a bright light and heard a loud scream"
    result = sensory_word_ratios(text)
    modality_sum = sum(result[sense] for sense in ('visual', 'auditory', 'tactile', 'olfactory', 'gustatory'))
    assert abs(result['total'] - modality_sum) < 1e-9
