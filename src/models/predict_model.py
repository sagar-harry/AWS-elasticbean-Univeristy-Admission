import joblib
import yaml
import numpy as np
import json


def retrieve_params(config):
    with open(config) as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params


params = retrieve_params(config="params.yaml")


def predict(data):
    cgpa_scaler = joblib.load(
        params["model_training"]["standard_scaler_location"]["cgpa_scaler"])
    gre_scaler = joblib.load(
        params["model_training"]["standard_scaler_location"]["gre_scaler"])
    model1 = joblib.load(params["model_training"]["model_location"])
    data["GRE Score"] = gre_scaler.transform(
        np.array(data["GRE Score"]).reshape(-1, 1))[0][0]
    data["CGPA"] = cgpa_scaler.transform(
        np.array(data["CGPA"]).reshape(-1, 1))[0][0]
    print(data)
    return model1.predict(np.array(list(data.values())).reshape(1, -1))[0]


def predict_value(data):
    with open(params["model_training"]["order_of_columns"], "r") as file:
        columns_names = json.loads(file.read())["Columns"]
    data1 = {}
    for i in columns_names:
        data1[i] = float(data[i])
    response = predict(data1)
    return float(response)*100
