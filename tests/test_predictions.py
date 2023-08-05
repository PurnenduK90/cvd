import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from sklearn.metrics import recall_score

from cvd_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_num_predictions = len(sample_input_data)

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    print(predictions)
    assert isinstance(predictions, np.ndarray ) #
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_num_predictions
    _predictions = list(predictions)
    y_true = sample_input_data["Heart_Disease"]
    recall = recall_score(_predictions, y_true)
    assert recall > 0.7