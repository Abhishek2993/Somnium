import train_distilbert as model_c


def test_is_enabled_defaults_to_false():
    assert model_c.is_enabled() is False


def test_fit_raises_when_disabled():
    try:
        model_c.fit(['some text'], ['SomeLabel'])
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert 'ENABLE_DISTILBERT' in str(e)
