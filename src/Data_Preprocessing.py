import os
import pandas as pd
import yaml
import argparse
from sklearn.model_selection import train_test_split
import mlflow


with open("./../config/config.yml", "r") as file:
    config = yaml.safe_load(file)

test_size = config['preprocess']['test_size']
Target = config['preprocess']['Target']
processed_data_path = config['data_paths']['processed_data_path']
clean_data_path = config['data_paths']['clean_data_path']

def processed_data(clean_data_path, processed_data_path, Target):
    print("#### preprocess Started####")
    clean_data_file = os.listdir(clean_data_path)[0]
    clean_data = os.path.join(clean_data_path, clean_data_file)
    df = pd.read_csv(clean_data)
    X = df.drop(columns=Target)
    Y = df[[Target]]
    X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = test_size)

    mlflow.log_param("test_size", test_size)

    X_train.to_csv(os.path.join(processed_data_path, "X_train.csv"))
    X_test.to_csv(os.path.join(processed_data_path, "X_test.csv"))
    y_train.to_csv(os.path.join(processed_data_path, "y_train.csv"))
    y_test.to_csv(os.path.join(processed_data_path, "y_test.csv"))
    print("##### Preprocessed Finished")

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_data_path", help="clean data path", default=clean_data_path)
    parser.add_argument("--processed_data_path", help="processed data path", default=processed_data_path)
    parser.add_argument("--Target", help = "provide Target", default=Target)
    args = parser.parse_args()
    processed_data(args.clean_data_path,args.processed_data_path,args.Target)

        