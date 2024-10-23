import pandas as pd
import argparse
import os
import yaml
import mlflow


with open("./../config/config.yml","r") as file:
#with open ("C:\mlops\prj\config\config.yml","r") as file:
    config = yaml.safe_load(file)


raw_data_path=config['data_paths']['raw_data_path']
clean_data_path = config['data_paths']['clean_data_path']

def data_cleaning(raw_data_path, clean_data_path):
    raw_data_file = os.listdir(raw_data_path)[0]
    print("####Data Cleaning Started###")
    raw_data = os.path.join(raw_data_path, raw_data_file)
    df = pd.read_csv(raw_data)

    clean_data_file = os.path.join(clean_data_path,raw_data_file)
    df.to_csv(clean_data_file, index=False)
    mlflow.log_param("cleaned_data_path", clean_data_file)
    print("#####Data Cleaning Finished#####")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", help="raw data path", default=raw_data_path)
    parser.add_argument("--clean_data_path", help="clean data path", default=clean_data_path)
    #parser.add_argument("--raw_data_file", help="provide raw data file name",default=raw_data_file)
    args = parser.parse_args()
    data_cleaning(args.raw_data_path, args.clean_data_path)