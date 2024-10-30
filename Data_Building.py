import os
import pandas as pd
import yaml
import argparse
from sklearn.linear_model import LinearRegression
import mlflow
import pickle
from sklearn.preprocessing import LabelEncoder

with open("./../config/config.yml", "r") as file:
    config = yaml.safe_load(file)

processed_data_path = config['data_paths']['processed_data_path']
model_path = config['data_paths']['model_path']
print(model_path)
fit_intercept = config['model']['fit_intercept']

def model_building(processed_data_path, model_path):
    print("#### Model Building Started ####")

    x_train_path = os.path.join(processed_data_path, "X_train.csv")
    y_train_path = os.path.join(processed_data_path, "y_train.csv")

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)

    #x_train_encoded = pd.get_dummies(x_train, drop_first=True)
    #print(f"Encoded X_train columns: {x_train_encoded.columns}")

    # Apply Label Encoding to y_train (Income_Category)
    #label_encoder = LabelEncoder()
    #y_train['Income_Category'] = label_encoder.fit_transform(y_train['Income_Category'])

    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(x_train,y_train)

    model_path_file = os.path.join(model_path, "linear_model.pkl")
    pickle.dump(model,open(model_path_file,"wb"))


    mlflow.log_param("fit_intercept", fit_intercept)
    mlflow.sklearn.log_model(model, "linear_reg")

    print(f"[INFO] model is exporeted")
    print("################MODEL BUILDING FINISHED#####################")

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_path", help="processed data path", default=processed_data_path)
    parser.add_argument("--model_path", help="model path", default=model_path)
    args = parser.parse_args()
    model_building(args.processed_data_path,args.model_path)
