import mlflow
import xgboost as xgb
import scipy

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from sklearn.metrics import mean_squared_error
from pathlib import Path
from preprocess import save_pickle, process_features, read_dataframe


def objective(params: dict) -> dict:

    data_path = Path("data")
    train_data_path = data_path.joinpath("green_tripdata_2022-01.parquet")
    valid_data_path = data_path.joinpath("green_tripdata_2022-02.parquet")

    dv, X_train, y_train = process_features(read_dataframe(train_data_path))
    X_val, y_val = process_features(read_dataframe(valid_data_path), dv) 
    

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)


    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=50,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}

def run_optimization():


    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )
    
    best_result["max_depth"] = int(best_result["max_depth"])

    return best_result

