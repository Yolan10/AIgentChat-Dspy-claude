import utils


def test_extract_simple():
    text = 'prefix [1, 2, 3] suffix'
    assert utils.extract_json_array(text) == [1, 2, 3]


def test_extract_nested():
    text = 'before [{"a": 1, "b": [2, 3]}, 4] after'
    assert utils.extract_json_array(text) == [{"a": 1, "b": [2, 3]}, 4]


def test_extract_invalid():
    text = 'no valid array here'
    assert utils.extract_json_array(text) is None


def test_extract_invalid_json():
    text = 'start [invalid json] end'
    assert utils.extract_json_array(text) is None


def test_extract_object_simple():
    text = 'prefix {"a": 1, "b": 2} suffix'
    assert utils.extract_json_object(text) == {"a": 1, "b": 2}


def test_extract_object_array():
    text = 'text before [1, 2, 3] after'
    assert utils.extract_json_object(text) == [1, 2, 3]


def test_extract_object_none():
    text = 'no json here'
    assert utils.extract_json_object(text) is None


def test_extract_size_limit():
    long_text = 'a' * (utils.MAX_JSON_SIZE + 1)
    assert utils.extract_json_array(long_text) is None
    assert utils.extract_json_object(long_text) is None
