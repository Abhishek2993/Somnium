from emotion import emotion_arc, sentence_emotion_scores, split_sentences, _linear_slope


def test_split_sentences_basic():
    assert split_sentences("I ran. Then I flew.") == ["I ran.", "Then I flew."]


def test_split_sentences_empty():
    assert split_sentences('') == []
    assert split_sentences('   ') == []


def test_sentence_emotion_scores_shape():
    scores = sentence_emotion_scores("I was so happy and joyful. Then everything turned dark and terrifying.")
    assert len(scores) == 2
    assert set(scores[0].keys()) == {'sentence', 'compound', 'pos', 'neg', 'neu'}


def test_sentence_emotion_scores_positive_vs_negative():
    scores = sentence_emotion_scores("This is wonderful and joyful. This is horrible and terrifying.")
    assert scores[0]['compound'] > 0
    assert scores[1]['compound'] < 0


def test_emotion_arc_empty_text():
    arc = emotion_arc('')
    assert arc == {'start': 0.0, 'end': 0.0, 'slope': 0.0, 'scores': []}


def test_emotion_arc_single_sentence_has_zero_slope():
    arc = emotion_arc("This was a wonderful dream.")
    assert arc['slope'] == 0.0
    assert arc['start'] == arc['end']


def test_emotion_arc_darkening_dream_has_negative_slope():
    text = "Everything was wonderful and joyful and happy. Then it got worse. Then it became horrible and terrifying."
    arc = emotion_arc(text)
    assert arc['slope'] < 0
    assert arc['start'] > arc['end']


def test_emotion_arc_brightening_dream_has_positive_slope():
    text = "It was horrible and terrifying. Then things improved. Then everything became wonderful and joyful."
    arc = emotion_arc(text)
    assert arc['slope'] > 0


def test_linear_slope_basic():
    assert _linear_slope([1.0, 2.0, 3.0]) == 1.0
    assert _linear_slope([3.0, 2.0, 1.0]) == -1.0
    assert _linear_slope([5.0]) == 0.0
    assert _linear_slope([]) == 0.0
