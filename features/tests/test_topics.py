import pytest

from topics import fit_topics, get_topic_labels

DOCS = [
    "I was flying over an enchanted castle with dragons and a wizard",
    "A mystical elf guided me through an ancient magical kingdom",
    "I discovered a fairy village hidden beneath a glowing forest",
    "A sorcerer turned the entire realm to gold with a flick of his wand",
    "A spy mission went wrong and I was being hunted by assassins",
    "I uncovered a government conspiracy and had to run before capture",
    "Dodging snipers I carried classified information through a war zone",
    "An undercover operation turned deadly when the betrayal was revealed",
]


def test_fit_topics_rejects_too_few_documents():
    with pytest.raises(ValueError):
        fit_topics(["one dream", "two dream"], min_topic_size=2)


@pytest.mark.slow
def test_fit_topics_and_get_topic_labels_on_small_corpus():
    topic_model, topic_ids = fit_topics(DOCS, min_topic_size=2, seed=42)
    assert len(topic_ids) == len(DOCS)
    labels = get_topic_labels(topic_model)
    assert isinstance(labels, dict)
    for words in labels.values():
        assert isinstance(words, list)
