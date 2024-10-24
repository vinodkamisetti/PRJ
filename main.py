import os
import mlflow
import yaml
import argparse


with open("./config/config.yml",'r') as file:
    config = yaml.safe_load(file)

with open("./config/secrets.yml", "r") as file:
    secrets = yaml.safe_load(file)

os.environ['MLFLOW_TRACKING_URI'] = secrets['mlflow']['MLFLOW_TRACKING_URI']
os.environ['MLFLOW_TRACKING_USERNAME'] = secrets['mlflow']['MLFLOW_TRACKING_USERNAME']
os.environ['MLFLOW_TRACKING_PASSWORD'] = secrets['mlflow']['MLFLOW_TRACKING_PASSWORD']

run_name = config['mlflow']['run_name']

mlflow.set_experiment("New")

def main(run_name):
    print("[INFO] MLOps pipeline triggerd")
    with mlflow.start_run(run_name=run_name):
        mlflow.run("./src", entry_point="Data_Cleaning.py", env_manager="local", run_name="Data_Cleaning")
        mlflow.run("./src", entry_point="Data_Preprocessing.py", env_manager="local", run_name="Data_Preprocessing")
        mlflow.run("./src", entry_point="Data_Building.py", env_manager="local", run_name="Data_Building")
        mlflow.run("./src", entry_point="Data_Evaluation.py", env_manager="local", run_name="Data_Evaluation")

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=run_name)
    args = parser.parse_args()
    main(args.run_name)