from typing import Dict, List


def get_predicted_tag(prediction: List[Dict[str, str]]) -> str:
    return prediction[0]["predicted_tags"]
