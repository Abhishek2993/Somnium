import pytest

from umap_embeddings import build_umap_figure, compute_umap_projection

TEXTS = [
    "I flew over an enchanted castle with dragons and a wizard",
    "A mystical elf guided me through an ancient magical kingdom",
    "A monster attacked me and I fought back in terror",
    "Ghosts screamed in the crumbling mansion as blood dripped down",
    "A spy mission went wrong and I was being hunted by assassins",
    "I uncovered a government conspiracy and had to run before capture",
]
LABELS = ['Fantasy', 'Fantasy', 'Horror', 'Horror', 'Thriller', 'Thriller']


@pytest.mark.slow
def test_compute_umap_projection_shape():
    projection = compute_umap_projection(TEXTS, n_neighbors=3, seed=42)
    assert projection.shape == (len(TEXTS), 2)


@pytest.mark.slow
def test_build_umap_figure_has_one_trace_per_label():
    fig = build_umap_figure(TEXTS, LABELS, seed=42)
    assert len(fig.data) == len(set(LABELS))
    total_points = sum(len(trace.x) for trace in fig.data)
    assert total_points == len(TEXTS)
