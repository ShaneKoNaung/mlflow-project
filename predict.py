import mlflow

from preprocess import read_dataframe, process_features,load_pickle
from pathlib import Path

from sklearn.metrics import mean_squared_error


MLFLOW_TRAKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRAKING_URI)


def predict_xgboost(run_id, features):

    logged_model = f'runs:/{run_id}/model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)


    predictions = loaded_model.predict(features)

    return predictions


if __name__ == "__main__":

    run_id = "fa6503255f8548d6bd25c8b76d076f13"

    valid_data_path = Path("data/green_tripdata_2022-03.parquet")
    pickle_file_path = Path("models/preprocessor.b")

    dv = load_pickle(pickle_file_path)

    X, y = process_features(read_dataframe(valid_data_path), dv)

    predictions = predict_xgboost(run_id, X)

    print(f"RMSE : {mean_squared_error(y, predictions)}")