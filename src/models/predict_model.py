import json

import joblib
import numpy as np
import yaml


def retrieve_params(config):
    with open(config) as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params


params = retrieve_params(config="params.yaml")


class InputException(Exception):
    def __init__(self, attribute, range):
        self.attribute = attribute
        self.range = range
        self.message = (
            f"{self.attribute} : Input values are not in specified range of {range}"
        )
        super().__init__(self.message)


def validate_input(data):
    schema_file = params["form_data"]["schema_json"]
    with open(schema_file) as json_file:
        schema_dict = json.load(json_file)

    for i in data.keys():
        if "min" in schema_dict[i].keys():
            if not schema_dict[i]["min"] <= data[i] <= schema_dict[i]["max"]:
                raise InputException(i, schema_dict[i])
        else:
            if not data[i] in schema_dict[i]["values"]:
                raise InputException(i, schema_dict[i])


def validate_output(output):
    schema_file = params["form_data"]["schema_json"]
    with open(schema_file) as json_file:
        schema_dict = json.load(json_file)

    if not schema_dict["output"]["min"] <= output <= schema_dict["output"]["max"]:
        raise Exception("Output not in range")


def predict(data):
    validate_input(data)
    cgpa_scaler = joblib.load(
        params["model_training"]["standard_scaler_location"]["cgpa_scaler"]
    )
    gre_scaler = joblib.load(
        params["model_training"]["standard_scaler_location"]["gre_scaler"]
    )
    model1 = joblib.load(params["prediction"]["final-model"])
    data["GRE Score"] = gre_scaler.transform(
        np.array(data["GRE Score"]).reshape(-1, 1)
    )[0][0]
    data["CGPA"] = cgpa_scaler.transform(np.array(data["CGPA"]).reshape(-1, 1))[0][0]
    response_ = model1.predict(np.array(list(data.values())).reshape(1, -1))[0]

    validate_output(response_)
    return response_


def predict_value(data):
    with open(params["model_training"]["order_of_columns"], "r") as file:
        columns_names = json.loads(file.read())["Columns"]
    data1 = {}
    for i in columns_names:
        data1[i] = float(data[i])
    response = predict(data1)
    return float(response) * 100
