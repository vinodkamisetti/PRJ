import mlflow
import os
import yaml
import argparse
from mlflow.tracking import MlflowClient

with open("./../config/config.yml", "r") as file:
    config = yaml.safe_load(file)

with open("./../config/secrets.yml", "r") as file:
    secrets = yaml.safe_load(file)

best_model_path = config['data_paths']['best_model_path']
experiment_name = config['mlflow']['experiment_name']

os.environ['MLFLOW_TRACKING_URI'] = secrets['mlflow']['MLFLOW_TRACKING_URI']
os.environ['MLFLOW_TRACKING_USERNAME'] = secrets['mlflow']['MLFLOW_TRACKING_USERNAME']
os.environ['MLFLOW_TRACKING_PASSWORD'] = secrets['mlflow']['MLFLOW_TRACKING_PASSWORD']

def model_fetching(best_model_path, experiment_name):
    print("#### Model Fetching Started ####")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    print(f"Experiment={experiment}")
    if experiment is None:
        raise ValueError(f"Experiment:{experiment_name} not found")
    experiment_id = experiment.experiment_id
    print(experiment_id)

    runs= client.search_runs(experiment_ids=[experiment_id], filter_string = "tags.mlflow.runName = 'Model_Evaluation'")
    print(runs)
    best_MAE = float("inf")
    best_r2_score=float("-inf")
    best_run=0
    for run in runs:
        metrics =  run.data.metrics
        MAE_test = metrics.get('MAE_test',float("inf"))
        r2_score_test = metrics.get('r2_score_test',float("inf"))

        if MAE_test < best_MAE and r2_score_test> best_r2_score:
            best_MAE = MAE_test
            best_r2_score = r2_score_test
            best_run = run

    if best_run:
        best_run_id = best_run.info.run_id
        print(f"Best run ID: {best_run_id} with MAE: {best_MAE} and R²: {best_r2_score}")

        model_uri = f"runs:/{best_run_id}/linear_reg"
        local_model_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=best_model_path)
                
        print(f"Model downloaded to: {local_model_path}")
        print("#################Model Fetching Finished################")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_model_path",help="provide raw data path", default=best_model_path)
    parser.add_argument("--experiment_name",help="provide cleaned data path", default=experiment_name)
    args = parser.parse_args()
    model_fetching(args.best_model_path,args.experiment_name)                        
