from nlp import fix_grammar, extract_keywords, get_named_entities, generate_title, GENRE_PREFIXES


def test_fix_grammar_capitalizes_and_punctuates():
    assert fix_grammar("i was flying over a castle") == "I was flying over a castle."


def test_fix_grammar_fixes_lowercase_i_mid_sentence():
    result = fix_grammar("then i saw a dragon")
    assert " I " in result or result.startswith("I ")


def test_fix_grammar_empty_string():
    assert fix_grammar("") == ""


def test_extract_keywords_filters_stopwords_and_short_words():
    keywords = extract_keywords("I dreamed I was flying over a giant golden castle with dragons")
    assert "golden" in keywords or "castle" in keywords
    assert "dream" not in keywords
    assert all(len(k) > 3 for k in keywords)


def test_extract_keywords_respects_top_n():
    text = "giant golden castle dragon wizard forest ocean mountain river valley desert jungle cave temple"
    keywords = extract_keywords(text, top_n=3)
    assert len(keywords) <= 3


def test_get_named_entities_finds_person_or_place():
    entities = get_named_entities("I met Albert Einstein in Paris while flying over the Eiffel Tower")
    labels = {e['label'] for e in entities}
    assert labels & {'PERSON', 'GPE', 'LOC', 'FAC'}


def test_generate_title_uses_known_prefix_for_genre():
    title = generate_title("I flew over an enchanted castle with a dragon", "Fantasy")
    prefix_words = {w for p in GENRE_PREFIXES["Fantasy"] for w in p.split()}
    assert any(word in title for word in prefix_words)


def test_generate_title_falls_back_for_unknown_genre():
    title = generate_title("A random event happened", "NotARealGenre")
    assert isinstance(title, str) and len(title) > 0
