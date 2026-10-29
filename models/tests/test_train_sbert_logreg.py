import pytest

import train_sbert_logreg as model_b

TEXTS = [
    "I flew over an enchanted castle with dragons and a wizard",
    "A mystical elf guided me through an ancient magical kingdom",
    "A monster attacked me and I fought back in terror",
    "Ghosts screamed in the crumbling mansion as blood dripped down",
]
LABELS = ['Fantasy', 'Fantasy', 'Horror', 'Horror']


@pytest.mark.slow
def test_fit_returns_expected_keys():
    fitted = model_b.fit(TEXTS, LABELS)
    assert set(fitted.keys()) == {'clf'}


@pytest.mark.slow
def test_predict_returns_one_label_per_text():
    fitted = model_b.fit(TEXTS, LABELS)
    predictions = model_b.predict(fitted, TEXTS)
    assert len(predictions) == len(TEXTS)
    assert all(p in LABELS for p in predictions)
