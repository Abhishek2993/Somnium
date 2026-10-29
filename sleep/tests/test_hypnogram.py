from hypnogram import STAGE_ORDER, hypnogram_series, plot_hypnogram, save_hypnogram


def test_hypnogram_series_shape_and_minutes():
    stages = ['W', 'N1', 'N2', 'N3', 'REM']
    series = hypnogram_series(stages, epoch_seconds=30.0)
    assert series['minutes'] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert len(series['stage_index']) == 5
    assert series['stage_order'] == STAGE_ORDER


def test_hypnogram_series_maps_stages_to_known_indices():
    series = hypnogram_series(['N3', 'W'])
    assert series['stage_index'][0] == STAGE_ORDER.index('N3')
    assert series['stage_index'][1] == STAGE_ORDER.index('W')


def test_hypnogram_series_unknown_stage_maps_to_negative_one():
    series = hypnogram_series(['W', 'NotAStage'])
    assert series['stage_index'][1] == -1


def test_plot_hypnogram_has_correct_number_of_points():
    stages = ['W', 'N1', 'N2', 'N2', 'N3', 'N3', 'REM', 'W']
    fig = plot_hypnogram(stages)
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == len(stages)
    assert len(fig.data[0].y) == len(stages)


def test_save_hypnogram_writes_html_file(tmp_path):
    out_path = str(tmp_path / 'sub' / 'hypnogram.html')
    save_hypnogram(['W', 'N1', 'N2', 'N3', 'REM', 'W'], out_path)
    with open(out_path) as f:
        content = f.read()
    assert '<html' in content.lower()
