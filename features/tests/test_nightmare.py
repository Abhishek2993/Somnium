from nightmare import classify_dream_type, nightmare_probability, train_nightmare_classifier


def test_nightmare_probability_neutral_for_empty_text():
    assert nightmare_probability('') == 0.5


def test_nightmare_probability_high_for_negative_aggressive_text():
    text = (
        "A monster attacked me and I fought back in terror. "
        "Everything was horrible and terrifying and I was trapped and injured."
    )
    assert nightmare_probability(text) > 0.5


def test_nightmare_probability_low_for_positive_friendly_text():
    text = (
        "My friends hugged me and we laughed together with joy. "
        "Everything was wonderful and I found a beautiful golden light."
    )
    assert nightmare_probability(text) < 0.5


def test_classify_dream_type_labels():
    nightmare_text = (
        "A monster attacked me and I fought back in terror. "
        "Everything was horrible and terrifying and I was trapped and injured."
    )
    pleasant_text = (
        "My friends hugged me and we laughed together with joy. "
        "Everything was wonderful and I found a beautiful golden light."
    )
    assert classify_dream_type(nightmare_text) == 'nightmare'
    assert classify_dream_type(pleasant_text) == 'pleasant'


def test_train_nightmare_classifier_fits_and_predicts():
    texts = [
        "A monster chased and attacked me and I was terrified",
        "I was trapped and injured and everything was horrible",
        "My friends hugged me and we laughed with joy",
        "I found a beautiful golden light and felt wonderful",
    ]
    labels = ['nightmare', 'nightmare', 'pleasant', 'pleasant']
    model = train_nightmare_classifier(texts, labels)
    predictions = model.predict(texts)
    assert list(predictions) == labels


def test_train_nightmare_classifier_rejects_mismatched_lengths():
    try:
        train_nightmare_classifier(['a', 'b'], ['nightmare'])
        assert False, "expected ValueError"
    except ValueError:
        pass
