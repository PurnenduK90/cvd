from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
import datetime
from cvd_model.config.core import config
from cvd_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    General_Health: Optional[str]
    Checkup: Optional[str]
    Exercise: Optional[str]
    Skin_Cancer: Optional[str]
    Other_Cancer: Optional[str]
    Depression: Optional[str]
    Diabetes: Optional[str]
    Arthritis: Optional[str]
    Sex: Optional[str]
    Age_Category: Optional[str]
    Height_cm: Optional[float]
    Weight_kg: Optional[float]
    BMI: Optional[float]
    Smoking_History: Optional[str]
    Alcohol_Consumption: Optional[float]
    Fruit_Consumption: Optional[float]
    Green_Vegetables_Consumption: Optional[float]
    FriedPotato_Consumption: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
