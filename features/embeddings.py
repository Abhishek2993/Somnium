"""
embeddings.py — shared sentence-embedding model loader, used by
bizarreness.py (Phase 3) and models/train_sbert_logreg.py (Phase 4) so
the model is only loaded once per process.
"""
import numpy as np

DEFAULT_MODEL_NAME = 'all-MiniLM-L6-v2'

_MODELS: dict = {}


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    if model_name not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS[model_name] = SentenceTransformer(model_name)
    return _MODELS[model_name]


def embed_sentences(sentences: list[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    if not sentences:
        return np.zeros((0, 0))
    model = get_embedding_model(model_name)
    return np.asarray(model.encode(sentences, convert_to_numpy=True))
