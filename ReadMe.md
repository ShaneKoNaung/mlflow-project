# Duration Prediction for NYC Taxi 

The purpose of this project is to practise MLOps Tools taught in DataTalks MlOps zoomcamp.

## Data

I am using the dataset from the NYC Taxi and Limousine Commission (TLC).
We are going to predict the duration of the ride for the green taxi. 

### Download the dataset
The dataset can be downloaded using **download_green_taxi_data** in preprocess.py.

**download_url** function from fastdownload lib is used to download the dataset.

```
year = 2022
month = 2
dest = Path("data/filename")

download_green_taxi_data(year, month, dest)

```

## Experiment Tracking using MLflow
For experiment tracking purpose, I am using mlflow. The very purpose of this porject is to practise MLflow for experiment tracking. 

Import MLflow model first.
```
import mlflow
```
In Mlflow, tracking data and artifacts can be stored 
    - locally in **mlruns** folder
    - using a SQLAlchemy-compatible database or 
    - using a tracking server

In this project, I am tracking the experiment locally using sqlite database.

After that, we can set the name of the experiment. MLflow will  create a new experiment if the experiment with this name doesn't exist.
If we don't set a name for experiment, mlflow will store data under default experiment.

```
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(experiment-name)
```

## MLflow UI and Server

If we are tracking experiments locally, we can use **mlflow ui** to access the dashboard.

```
#bash

# locally
mlflow ui 

# locally with sqlite database
mlflow ui --backend-store-uri=sqlite:///mlflow.db

# using mlflow server
mlflow server --backend-store-uri=sqlite:///mlflow.db

```

### Tracking runs using MLflow

#### mlflow.start_run()

We can track specific data for each using **mlflow.start_run()**. We can set tags and log metric, param, artifacts and models.

```
with mlflow.start_run():
    # training code
    ...

```

#### Setting Tags 

We can also assign tags to the run by using **set_tag**.

```
mlflow.set_tag("developer", "shane")
```

#### Logging Parameters

We can log parameters using **log_param** or **log_params**.
```
mlflow.log_param({"alpha", 0.1})

params = {
        'learning_rate': 0.09585355369315604,
        'max_depth': 30,
        'min_child_weight': 1.060597050922164,
        'objective': 'reg:linear',
        'reg_alpha': 0.018060244040060163,
        'reg_lambda': 0.011658731377413597,
        'seed': 42
    }

mlflow.log_params(params)
```

#### log metric

We can also log metrics using **log_metric**.
```
mlflow.log_metric("rmse", rmse)
```

#### log artifacts

We can log a single artifact using  **log_artifact** and we can log a folder of artifacts using **log_artifacts**.

```
mlflow.log_artifact("models/preprocessor.b", artifact_path="artifacts")


mlflow.log_artifacts("models", artifact_path="artifacts")

```

#### logging models

here is how we can log models for specific lib.

```
# for sklearn model

from mlflow.models.signature import infer_signature

signature = infer_signature(X, model.predict(X))
mlflow.sklearn.log_model(model, artifact_path="models", signature=signature)


# for xgboost model
signature = infer_signature(X_test, y_preds)
mlflow.xgboost.log_model(booster, artifact_path="model", signature=signature)

```

We can see the whole thing here. 
```
def objective(params: dict) -> dict:

    # feature processing
    ...


    with mlflow.start_run():

        
        mlflow.set_tag("Developer", "Shane")
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

        signature = infer_signature(X_test, y_preds)
        mlflow.xgboost.log_model(booster, artifact_path="model", signature=signature)
```

### Auto Logging
We can also use auto logging for tracking. 
MLflow supports autolog for various machine learning and deep learning libs.

```
mlflow.<framework>.autolog()
```

In order to use auto log for sklearn model, we can use **mlflow.sklearn.autolog()**.

```
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


    return lr
```



## Train

train.py is used for training the model.

```
# Train Lasso model using scikit-learn lib
python train.py lasso 

# Train xgboost model using Xgboost lib
python train.py xgboost

# Train xgboost model after performing hypyerparameter tuning using hyperopt lib
python train.py xgboost_hopt

# Train both Lasso model and xgboost model with hyperparameter tuning
python train.py all
```


# Model Loading

For loading model, we only need the run_id of the model.

```
import mlflow

MLFLOW_TRAKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRAKING_URI)

run_id = "fa6503255f8548d6bd25c8b76d076f13"

logged_model = f'runs:/{run_id}/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# preprocessing features
X_test = ...

predictions = loaded_model.predict(X_test)
```


# Model Register

## Searching Experiments and Runs

The mlflow.client module provides a Python CRUD interface to MLflow Experiments, Runs, Model Versions, and Registered Models. 

### Search Experiments

```
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

client.search_experiments()

out : [ <Experiment: artifact_location='/home/shane/mlflow-project/mlruns/1', creation_time=1687998653695, experiment_id='1', last_update_time=1687998653695, lifecycle_stage='active', name='NYC-Green-Taxi', tags={}>,
        <Experiment: artifact_location='/home/shane/mlflow-project/mlruns/0', creation_time=1687998653682, experiment_id='0', last_update_time=1687998653682, lifecycle_stage='active', name='Default', tags={}>]
```

### Search runs

We can search runs for each experiment using the experiment id.
The runs can be filtered and ordered using filter_string and order_by.
```
from mlflow.entities import ViewType

experiment_id = "1"

runs = client.search_runs(
                experiment_ids=experiment_id,
                filter_string="metrics.rmse < 6.0",
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=5,
                order_by=["metrics.rmse ASC"])


for run in runs:
    print(f"run_name : {run.info.run_name}")
    print(f"run_id : {run.info.run_id}")
    print(f"rmse : {run.data.metrics['rmse']}")

out :   run_name : serious-finch-887
        run_id : fa6503255f8548d6bd25c8b76d076f13
        rmse : 5.818090066449871
```

## Register Model

mlflow.register_model will create a new model if 'model_name' doesn't exist.
It will register the model under the model_name.

```
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

run_id = "fa6503255f8548d6bd25c8b76d076f13"
model_uri = f"runs:/{run_id}/model"

model_name = "nyc-taxi-duration-prediction"

mlflow.register_model(model_uri=model_uri, name=model_name)

```

## Model Stage Transition

We can change the stage tag of the modelusing transition_model_version_stage.
We can assign "Staging", "Production or "Archive".

```
model_version = version.version
new_stage = "Staging"

client.transition_model_version_stage(
                    name=model_name,
                    version=model_version,
                    stage=new_stage,
                    archive_existing_versions=False)


out : <ModelVersion: aliases=[], creation_timestamp=1688006850748, current_stage='Staging', description=None, last_updated_timestamp=1688006967170, name='nyc-taxi-duration-prediction', run_id='fa6503255f8548d6bd25c8b76d076f13', run_link=None, source='/home/shane/mlflow-project/mlruns/1/fa6503255f8548d6bd25c8b76d076f13/artifacts/model', status='READY', status_message=None, tags={}, user_id=None, version=1>

```

## Add Model Description

In this example, we can add description to the registered model.

```
from datetime import datetime

date = datetime.today().date()

client.update_model_version(
        name=model_name,
        version=model_version,
        description=f"The model version {model_version} was transition to {new_stage} on {date}")
```

## References

- datatalks mlops zoomcamp mlflow chapter - https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/02-experiment-tracking
- mlflow docs - https://mlflow.org/docs/latest/index.html
- TLC Trip Record Data - https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- fastdownload docs - https://fastdownload.fast.ai/