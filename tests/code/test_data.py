import pytest
from tagifai import data


@pytest.mark.parametrize(
    "text, sw, clean_text",
    [("hi", [], "hi"), ("hi you", ["you"], "hi"), ("hi yous", ["you"], "hi yous")],
)
def test_clean_text(text, sw, clean_text):
    assert data.clean_text(text=text, stopwords=sw) == clean_text
