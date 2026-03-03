import pytest
import pandas as pd
from data.validation import validate_data

def test_validate_data_good():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert validate_data(df) is True

def test_validate_data_empty():
    df = pd.DataFrame({"a": [], "b": []})
    assert validate_data(df) is False

def test_validate_data_nulls():
    df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
    assert validate_data(df) is False

def test_validate_data_list_of_dicts():
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert validate_data(data) is True

def test_validate_data_bad_type():
    assert validate_data("not a dataframe") is False

def test_validate_data_none():
    assert validate_data(None) is False
