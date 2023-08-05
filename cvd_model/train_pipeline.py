import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from cvd_model.config.core import config
from cvd_model.pipeline import pipe
from cvd_model.processing.data_manager import load_dataset, save_pipeline

def run_training(debug=False) -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting
    pipe.fit(X_train,y_train)  #
    now = time.time()
    y_pred = pipe.predict(X_test)
    itime = time.time() - now
    speed = len(y_pred)/itime
    report = classification_report(y_test, y_pred, output_dict=True)
    if debug:
        print("#"*55)
        print(f"Accuracy = {report['accuracy']:0.02f} %, Inference = {speed:0.02f} predictions/sec")
        del report['accuracy']
        report_df = pd.DataFrame(report).T
        print("_"*55)
        print(report_df)
        print("_"*55) 
    # persist trained model
    save_pipeline(pipeline_to_persist= pipe)
    # printing the score
    
if __name__ == "__main__":
    debug = False
    if len(sys.argv) > 1:
        dbg_arg = sys.argv[1].lower().strip().replace(" ", "")
        if dbg_arg in ["debug=true", "debug=yes"]:
            debug = True
    run_training(debug)