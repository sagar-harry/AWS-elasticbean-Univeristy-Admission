import argparse
import json

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow

from urllib.parse import urlparse


def retrieve_params(config):
    with open(config) as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params


def trainTestSplit(params):
    df = pd.read_csv(params["data"]["processed_data"])
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1],
        df.iloc[:, -1],
        train_size=params["model_training"]["splitting_params"]["split_size"],
        random_state=params["model_training"]["splitting_params"]["random_state"],
    )

    return X_train, X_test, y_train, y_test


def standardizing(X_train, X_test, params):
    gre_scaler = StandardScaler()
    cgpa_scaler = StandardScaler()
    X_train["GRE Score"] = gre_scaler.fit_transform(
        np.array(X_train["GRE Score"]).reshape(-1, 1)
    )
    X_test["GRE Score"] = gre_scaler.transform(
        np.array(X_test["GRE Score"]).reshape(-1, 1)
    )

    X_train["CGPA"] = cgpa_scaler.fit_transform(
        np.array(X_train["CGPA"]).reshape(-1, 1)
    )
    X_test["CGPA"] = cgpa_scaler.transform(np.array(X_test["CGPA"]).reshape(-1, 1))

    joblib.dump(
        gre_scaler,
        params["model_training"]["standard_scaler_location"]["gre_scaler"],
        compress=True,
    )
    joblib.dump(
        cgpa_scaler,
        params["model_training"]["standard_scaler_location"]["cgpa_scaler"],
        compress=True,
    )
    with open(params["model_training"]["order_of_columns"], "w") as file:
        file.write(json.dumps({"Columns": list(X_train.columns)}))
    return X_train, X_test


def train_model(config):
    params = retrieve_params(config=config)

    X_train, X_test, y_train, y_test = trainTestSplit(params)
    X_train_standardized, X_test_standardized = standardizing(X_train, X_test, params)

    mlflow_config = params["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        alpha = alpha=params["model_training"]["model_params"]["Ridge"]["alpha"]
        model1 = Ridge(alpha)
        model1.fit(X_train_standardized, y_train)

        print(r2_score(y_train, model1.predict(X_train_standardized)))
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("r2_score", r2_score(y_test, model1.predict(X_test_standardized)))

        with open(config, "r") as my_file:
            my_dict = yaml.safe_load(my_file)
            my_dict["model_training"]["model_metrics"]["testing_r2_score"] = float(
                r2_score(y_test, model1.predict(X_test_standardized))
            )
            my_dict["model_training"]["model_metrics"]["training_r2_score"] = float(
                r2_score(y_train, model1.predict(X_train_standardized))
            )

        with open(config, "w") as my_file2:
            yaml.dump(my_dict, my_file2, allow_unicode=True, indent=4)

        params = retrieve_params(config=config)
        joblib.dump(model1, params["model_training"]["model_location"])

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(model1, "model", registered_model_name=mlflow_config["registered_model_name"])
        
        else:
            mlflow.sklearn.log_model(model1, "model")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_model(parsed_args.config)

