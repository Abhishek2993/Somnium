"""
umap_embeddings.py — interactive UMAP (Plotly HTML) of dream sentence
embeddings, colored by genre. Needs `umap-learn` and `plotly`.
"""
import os

import paths  # noqa: F401

from embeddings import embed_sentences


def compute_umap_projection(texts: list[str], n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42):
    from umap import UMAP

    embeddings = embed_sentences(texts)
    n = len(texts)
    reducer = UMAP(
        n_neighbors=min(n_neighbors, max(2, n - 1)),
        min_dist=min_dist,
        n_components=2,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def build_umap_figure(texts: list[str], labels: list[str], embedding_2d=None, seed: int = 42):
    import plotly.graph_objects as go

    if embedding_2d is None:
        embedding_2d = compute_umap_projection(texts, seed=seed)

    fig = go.Figure()
    for label in sorted(set(labels)):
        indices = [i for i, l in enumerate(labels) if l == label]
        fig.add_trace(go.Scatter(
            x=[embedding_2d[i, 0] for i in indices],
            y=[embedding_2d[i, 1] for i in indices],
            mode='markers',
            name=label,
            text=[texts[i] for i in indices],
            hoverinfo='text+name',
        ))
    fig.update_layout(
        title='UMAP of dream sentence embeddings (synthetic genre corpus)',
        xaxis_title='UMAP-1', yaxis_title='UMAP-2',
    )
    return fig


def main():
    import ml

    texts, labels = ml.get_training_corpus()
    print(f'Computing UMAP projection for {len(texts)} texts ({len(set(labels))} genres)...')
    fig = build_umap_figure(texts, labels)

    os.makedirs(paths.REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(paths.REPORTS_DIR, 'umap_embeddings.html')
    fig.write_html(out_path)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
