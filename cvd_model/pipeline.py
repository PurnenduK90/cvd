import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from imblearn.pipeline import Pipeline as impipe
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from cvd_model.config.core import config
from cvd_model.processing.features import numeric
from cvd_model.processing.features import categorical
from cvd_model.processing.features import age
from cvd_model.processing.features import genhealth
from cvd_model.processing.features import checkup

num_features = config.model_config.numerical_features
cat_features = config.model_config.categorical_features.drop([
    config.model_config.genhealth_var,
    config.model_config.age_var,
    config.model_config.checkup.var
])

preprocessing = ColumnTransformer(transformers=[
    ("numerical", numeric, num_features),
    ("categorical", categorical, cat_features),
    ("age", age, [config.model_config.age_var]),
    ("general", genhealth, [config.model_config.genhealth_var]),
    ("checkup", checkup, [config.model_config.checkup_var])
])

pipe=impipe(
    steps=[
        ('pre', preprocessing),
        ('under', RandomUnderSampler(random_state=config.model_config.random_state)),
        ('grid', GradientBoostingClassifier(
            random_state=config.model_config.random_state,
            learning_rate=config.model_config.learning_rate,
            max_depth=config.model_config.max_depth,
            n_estimators=config.model_config.n_estimators
        )
    )
])