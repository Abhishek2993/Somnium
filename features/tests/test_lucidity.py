from lucidity import lucid_probability, train_lucid_classifier


def test_lucid_probability_zero_for_empty_text():
    assert lucid_probability('') == 0.0


def test_lucid_probability_zero_for_ordinary_dream():
    assert lucid_probability("I was walking through a forest and saw a deer") == 0.0


def test_lucid_probability_positive_for_single_lucid_phrase():
    score = lucid_probability("I realized that I was dreaming and kept walking")
    assert 0.0 < score <= 1.0


def test_lucid_probability_full_for_multiple_lucid_phrases():
    text = "I realized I was dreaming. It was a lucid dream and I took control of the dream."
    assert lucid_probability(text) == 1.0


def test_train_lucid_classifier_fits_and_predicts():
    texts = [
        "I realized I was dreaming and took control of the dream",
        "It was a lucid dream and I flew wherever I wanted",
        "I walked to the store and bought some bread",
        "I was late for work and missed the bus",
    ]
    labels = [True, True, False, False]
    model = train_lucid_classifier(texts, labels)
    predictions = model.predict(texts)
    assert list(predictions) == labels


def test_train_lucid_classifier_rejects_mismatched_lengths():
    try:
        train_lucid_classifier(['a', 'b'], [True])
        assert False, "expected ValueError"
    except ValueError:
        pass
