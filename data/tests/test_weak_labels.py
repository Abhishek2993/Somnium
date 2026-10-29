from weak_labels import weak_is_lucid, weak_is_nightmare, weak_label


def test_weak_is_lucid_true_for_luciddreaming_subreddit():
    assert weak_is_lucid('LucidDreaming', None) is True


def test_weak_is_lucid_true_for_lucid_flair_elsewhere():
    assert weak_is_lucid('dreams', 'Lucid Dream') is True


def test_weak_is_lucid_false_otherwise():
    assert weak_is_lucid('dreams', 'Nightmare') is False
    assert weak_is_lucid('dreams', None) is False


def test_weak_is_nightmare_true_for_nightmares_subreddit():
    assert weak_is_nightmare('Nightmares', None) is True


def test_weak_is_nightmare_true_for_nightmare_flair_elsewhere():
    assert weak_is_nightmare('dreams', 'Night Terror') is True


def test_weak_is_nightmare_false_otherwise():
    assert weak_is_nightmare('dreams', 'Lucid') is False
    assert weak_is_nightmare('LucidDreaming', None) is False


def test_weak_label_combines_both_flags():
    assert weak_label('dreams', None) == {'weak_lucid': False, 'weak_nightmare': False}
    assert weak_label('Nightmares', None) == {'weak_lucid': False, 'weak_nightmare': True}
    assert weak_label('LucidDreaming', None) == {'weak_lucid': True, 'weak_nightmare': False}


def test_weak_label_case_insensitive_subreddit():
    assert weak_is_lucid('luciddreaming', None) is True
    assert weak_is_nightmare('NIGHTMARES', None) is True
