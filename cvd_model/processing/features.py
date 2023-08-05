from typing import List, Union
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
#import datetime as dt

from sklearn.base import BaseEstimator, TransformerMixin

numeric = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore',drop='first'))
])
    
## Age Category Pipeline
age = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder",OrdinalEncoder())
])

## General Health Pipeline
genhealth = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Fair")),
    ("encoder",OrdinalEncoder(categories=[['Poor','Fair','Good','Very Good','Excellent']]))
])

## Checkup Pipeline
checkup = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value='Within the past 5 years')),
    ("encoder",OrdinalEncoder(categories=[['Within the past year','Within the past 2 years','Within the past 5 years','5 or more years ago','Never']]))
])