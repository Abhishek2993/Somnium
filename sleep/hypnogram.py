"""
hypnogram.py — plots a sequence of per-epoch sleep stages as a
hypnogram (Plotly HTML), the standard step-chart view of a night's
sleep stages over time.
"""
import os

# Bottom to top: N3 (deepest) at the bottom, wake at the top -- the
# conventional hypnogram layout.
STAGE_ORDER = ['N3', 'N2', 'N1', 'REM', 'W']


def hypnogram_series(stage_labels: list[str], epoch_seconds: float = 30.0) -> dict:
    """{'minutes': [...], 'stage_index': [...], 'stage_order': STAGE_ORDER}."""
    stage_to_index = {stage: i for i, stage in enumerate(STAGE_ORDER)}
    minutes = [i * epoch_seconds / 60.0 for i in range(len(stage_labels))]
    stage_index = [stage_to_index.get(stage, -1) for stage in stage_labels]
    return {'minutes': minutes, 'stage_index': stage_index, 'stage_order': STAGE_ORDER}


def plot_hypnogram(stage_labels: list[str], epoch_seconds: float = 30.0):
    import plotly.graph_objects as go

    series = hypnogram_series(stage_labels, epoch_seconds=epoch_seconds)
    fig = go.Figure(go.Scatter(
        x=series['minutes'], y=series['stage_index'],
        mode='lines', line_shape='hv', line=dict(width=2),
    ))
    fig.update_yaxes(
        tickvals=list(range(len(STAGE_ORDER))), ticktext=STAGE_ORDER,
        title='Sleep stage',
    )
    fig.update_xaxes(title='Time (minutes)')
    fig.update_layout(title='Hypnogram')
    return fig


def save_hypnogram(stage_labels: list[str], out_path: str, epoch_seconds: float = 30.0) -> None:
    fig = plot_hypnogram(stage_labels, epoch_seconds=epoch_seconds)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
