"""
topics.py — theme topics across a corpus of dreams, via BERTopic.

bertopic/umap/hdbscan are only imported inside fit_topics() (not at
module load time), so importing this module -- and the rest of the
features/ package -- doesn't require them to be installed unless this
function is actually called. BERTopic needs a reasonably sized corpus
to find stable topics; with 0 real dreams collected so far (see
data/README.md) this has only been exercised on small synthetic
corpora in features/tests/test_topics.py, not on real dream text.
"""
MIN_DOCS_PER_TOPIC = 2


def fit_topics(texts: list[str], min_topic_size: int = MIN_DOCS_PER_TOPIC, seed: int = 42):
    """
    Fit a BERTopic model over `texts`. Returns (topic_model, topic_ids)
    where topic_ids[i] is the topic assigned to texts[i] (-1 = outlier).
    Raises ValueError if there aren't enough documents to form even two
    topics of min_topic_size.
    """
    if len(texts) < min_topic_size * 2:
        raise ValueError(
            f'Need at least {min_topic_size * 2} documents to fit topics, got {len(texts)}'
        )

    from bertopic import BERTopic
    from umap import UMAP

    n = len(texts)
    umap_model = UMAP(
        random_state=seed,
        n_neighbors=min(15, max(2, n - 1)),
        n_components=min(5, max(2, n - 1)),
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
    )
    topic_ids, _ = topic_model.fit_transform(texts)
    return topic_model, list(topic_ids)


def get_topic_labels(topic_model, top_n_words: int = 5) -> dict:
    """{topic_id: [top words...]} for every non-outlier topic found."""
    labels = {}
    for topic_id in topic_model.get_topics():
        if topic_id == -1:
            continue
        words = [word for word, _ in topic_model.get_topic(topic_id)[:top_n_words]]
        labels[topic_id] = words
    return labels
