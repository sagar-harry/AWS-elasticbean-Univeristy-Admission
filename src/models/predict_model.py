import argparse
import joblib
import yaml
import numpy as np

def retrieve_params(config):
    with open(config) as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params

def predict(config, data):
    params = retrieve_params(config=config)
    cgpa_scaler = joblib.load(params["model_training"]["standard_scaler_location"]["cgpa_scaler"])
    gre_scaler = joblib.load(params["model_training"]["standard_scaler_location"]["gre_scaler"])
    model1 = joblib.load(params["model_training"]["model_location"])
    data["GRE_score"] = gre_scaler.transform(data["GRE_score"].reshape(-1,1))[0][0]
    data["CGPA"] = cgpa_scaler.transform(data["CGPA"].reshape(-1,1))[0][0]
    return model1.predict(data.values())

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    args.add_argument("--data")
    parsed_args = args.parse_args()
    predict(config=parsed_args.config, data=parsed_args.data)
