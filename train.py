from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from preprocess import check_data

import scipy
import xgboost as xgb
import mlflow


def train_linear_sklearn(model : linear_model ,X_train : scipy.sparse._csr.csr_matrix, 
                         y_train : scipy.sparse._csr.csr_matrix, 
                         X_val : scipy.sparse._csr.csr_matrix, 
                         y_val : scipy.sparse._csr.csr_matrix) -> linear_model:
    
    with mlflow.start_run() as run:

        mlflow.set_tag("Developer", "Shane")

        lr = model()

        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)

    return lr


def train_xgboost(X_train : scipy.sparse._csr.csr_matrix, 
                y_train : scipy.sparse._csr.csr_matrix, 
                X_val : scipy.sparse._csr.csr_matrix, 
                y_val : scipy.sparse._csr.csr_matrix,
                params: dict) -> xgb.core.Booster:

    with mlflow.start_run() as run:

        mlflow.set_tag("Developer", "Shane")


        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_train)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_round=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)

    return booster


if __name__ == "__main__":

    year = args.year
    month = args.month

    if check_data()