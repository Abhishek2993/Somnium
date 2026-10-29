import dreambank_loader as dl

DREAM_A = (
    "I was walking through a long hallway that kept stretching further away "
    "every time I took a step forward, and the lights above me flickered in "
    "a strange rhythm while I tried to find a door that would let me out of "
    "this endless corridor before something behind me caught up with me."
)
DREAM_B = (
    "A friend from childhood appeared at the edge of a lake and told me the "
    "water was safe to swim in, so I waded in slowly and noticed the whole "
    "lake was lined with small floating lanterns that glowed brighter the "
    "deeper I went, until the entire shoreline was a ring of golden light."
)


def test_split_into_dreams_on_blank_lines():
    raw = f"{DREAM_A}\n\n{DREAM_B}\n"
    dreams = dl.split_into_dreams(raw)
    assert len(dreams) == 2
    assert dreams[0].startswith('I was walking')
    assert dreams[1].startswith('A friend')


def test_split_into_dreams_empty_input():
    assert dl.split_into_dreams('') == []


def test_strip_header_removes_numeric_header_line():
    paragraph = f"1.\n{DREAM_A}"
    assert dl.strip_header(paragraph) == DREAM_A


def test_strip_header_removes_date_header_line():
    paragraph = f"3/14/1985\n{DREAM_A}"
    assert dl.strip_header(paragraph) == DREAM_A


def test_strip_header_leaves_text_without_header_unchanged():
    assert dl.strip_header(DREAM_A) == DREAM_A


def test_load_dreambank_file_parses_and_filters(tmp_path):
    path = tmp_path / 'series.txt'
    path.write_text(f"1.\n{DREAM_A}\n\n2.\ntoo short\n\n3.\n{DREAM_B}\n", encoding='utf-8')

    records = dl.load_dreambank_file(str(path), min_words=10)

    assert len(records) == 2
    texts = {r['text'] for r in records}
    assert DREAM_A in texts
    assert DREAM_B in texts
    assert all(r['source_label'] == 'series.txt' for r in records)
    assert all(r['content_hash'] for r in records)
