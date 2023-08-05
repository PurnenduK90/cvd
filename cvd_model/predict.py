import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Union
import pandas as pd

from cvd_model import __version__ as _version
from cvd_model.config.core import config
from cvd_model.pipeline import pipe
from cvd_model.processing.data_manager import load_pipeline
from cvd_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    indexed_data= validated_data.reindex(columns=config.model_config.features)

    results = {"predictions": None,"version": _version, "errors": errors}
    if not errors:

        predictions = pipe.predict(indexed_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results