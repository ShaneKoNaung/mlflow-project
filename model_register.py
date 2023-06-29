from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow


def list_experiments(client):
    exps = client.search_experiments()

    return exps

def list_runs(client, experiment_ids, 
              filter_string, 
              max_results, order_by):
    
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=max_results,
        order_by=order_by
    )

    for run in runs:
        print(f"run_name : {run.info.run_name}")
        print(f"run_id : {run.info.run_id}")
        print(f"rmse : {run.data.metrics['rmse']}")

    return runs

def register_model(run_id, model_name):
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=model_name)
    

def transition_model_version_stage(client,model_name, model_version, new_stage,
                                archive_existing_versions):
    client.transition_model_version_stage(
                    name=model_name,
                    version=model_version,
                    stage=new_stage,
                    archive_existing_versions=False)
    

