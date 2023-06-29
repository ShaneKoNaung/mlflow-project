from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from pathlib import Path

from preprocess import read_dataframe, process_features, save_pickle

from hopt import run_optimization

from argparse import ArgumentParser

import scipy
import xgboost as xgb
import mlflow
from mlflow.models.signature import infer_signature



def train_linear_sklearn(model : linear_model ,X_train : scipy.sparse._csr.csr_matrix, 
                         y_train : scipy.sparse._csr.csr_matrix, 
                         X_val : scipy.sparse._csr.csr_matrix, 
                         y_val : scipy.sparse._csr.csr_matrix) -> linear_model:
    
    mlflow.sklearn.autolog()
    with mlflow.start_run() as run:

        mlflow.set_tag("Developer", "Shane")

        print("Training Model")

        lr = model()

        lr.fit(X_train, y_train)

        print("Model Training Done")

        y_pred = lr.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

        print(f"RMSE : {rmse}")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="artifacts")

        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.log_model(lr, artifact_path="models", signature=signature)


    return lr


def train_xgboost(X_train : scipy.sparse._csr.csr_matrix, 
                y_train : scipy.sparse._csr.csr_matrix, 
                X_val : scipy.sparse._csr.csr_matrix, 
                y_val : scipy.sparse._csr.csr_matrix,
                params: dict = {}) -> xgb.core.Booster:
    mlflow.xgboost.autolog()
    with mlflow.start_run() as run:

        mlflow.set_tag("Developer", "Shane")


        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)


        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="artifacts")

        print(f"Xgboost RMSE : {rmse}")

        signature = infer_signature(X_val, y_pred)
        mlflow.xgboost.log_model(booster, artifact_path="models", signature=signature)

    return booster


if __name__ == "__main__":

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("NYC-Green-Taxi")

    parser = ArgumentParser()
    parser.add_argument("model", choices=["lasso", "xgboost", "xgboost_hopt", "all"])
    args = parser.parse_args()

    model_type = args.model
    
    data_path = Path("data")
    train_data_path = data_path.joinpath("green_tripdata_2022-01.parquet")
    valid_data_path = data_path.joinpath("green_tripdata_2022-02.parquet")

    model_path = Path("models")
    dv_path = model_path.joinpath("preprocessor.b")

    if not model_path.exists():
        model_path.mkdir()

    dv, X_train, y_train = process_features(read_dataframe(train_data_path))
    save_pickle(dv, dv_path)
    X_val, y_val = process_features(read_dataframe(valid_data_path), dv)



    if model_type == "lasso" or model_type == "all":
        model = linear_model.Lasso  

        model = train_linear_sklearn(model, X_train, y_train, X_val, y_val)

    if model_type == "xgboost":

        model = train_xgboost(X_train, y_train, X_val, y_val)
    
    if model_type == "xgboost_hopt" or model_type == "all":
        best_result = run_optimization()
        print(best_result)
        model = train_xgboost(X_train, y_train, X_val, y_val, best_result)
        