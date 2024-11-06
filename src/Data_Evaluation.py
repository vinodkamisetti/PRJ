 import os
import pandas as pd
import yaml
import argparse
from sklearn.model_selection import train_test_split
import mlflow
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

with open("./../config/config.yml", "r") as file:
    config = yaml.safe_load(file)

processed_data_path = config['data_paths']['processed_data_path']
model_path = config['data_paths']['model_path']

def model_eval(processed_data_path, model_path):
    print("#### Model Evaluation Started ####")
    model_path_file = os.path.join(model_path, "linear_model.pkl")
    model = pickle.load(open(model_path_file, 'rb'))

    x_train_path = os.path.join(processed_data_path, "X_test.csv")
    y_train_path = os.path.join(processed_data_path, "y_test.csv")

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)

    y_pred_train = model.predict(x_train)

    MAE_train = mean_absolute_error(y_train,y_pred_train)
    MSE_train = mean_squared_error(y_train, y_pred_train)
    r2_score_train = r2_score(y_train, y_pred_train)

    #mlflow.log_metric({"MAE_train": MAE_train, "MSE_train": MSE_train, "r2_score_train": r2_score_train})
    mlflow.log_metric("MAE_train", MAE_train)
    mlflow.log_metric("MSE_train", MSE_train)
    mlflow.log_metric("r2_score_train", r2_score_train)

    mlflow.sklearn.log_model(model, "linear.reg")
    print("##### model Evalution Finished ####")

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_path", help="processed data path", default=processed_data_path)
    parser.add_argument("--model_path", help="model path", default=model_path)
    args = parser.parse_args()
    model_eval(args.processed_data_path,args.model_path)

    
        
