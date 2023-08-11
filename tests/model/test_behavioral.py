import pytest
from tagifai import main
from utils import get_predicted_tag


@pytest.mark.parametrize(
    "input_a, input_b, tag",
    [
        (
            "Transformers applied to NLP have revolutionized machine learning.",
            "Transformers applied to NLP have disrupted machine learning.",
            "natural-language-processing",
        )
    ],
)
def test_invariance(input_a, input_b, tag, run_id):
    """Invariance via verb injection should not affect outputs"""
    tag_a = get_predicted_tag(main.predict_tag(input_a, run_id))
    tag_b = get_predicted_tag(main.predict_tag(input_b, run_id))
    assert tag_a == tag_b == tag


@pytest.mark.parametrize(
    "input, tag",
    [
        (
            "ML applied to text classification.",
            "natural-language-processing",
        ),
        (
            "ML applied to image classification.",
            "computer-vision",
        ),
        (
            "CNNs for text classification.",
            "natural-language-processing",
        ),
    ],
)
def test_directional(input, tag, run_id):
    """Directional expectations (changes with known outputs)."""
    prediction = get_predicted_tag(main.predict_tag(input, run_id))
    assert tag == prediction


@pytest.mark.parametrize(
    "input, tag",
    [
        (
            "Natural language processing is the next big wave in machine learning.",
            "natural-language-processing",
        ),
        (
            "MLOps is the next big wave in machine learning.",
            "mlops",
        ),
        (
            "This is about graph neural networks.",
            "other",
        ),
    ],
)
def test_mft(input, tag, run_id):
    """Minimum Functionality Tests (simple input/output pairs)."""
    prediction = get_predicted_tag(main.predict_tag(input, run_id))
    assert tag == prediction
