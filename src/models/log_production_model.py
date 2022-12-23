from http import client
import argparse
import joblib
import yaml

import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint

def retrieve_params(config):
    with open(config) as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params

def log_production_model(config_path):
    config = retrieve_params(config=config_path)
    mlflow_config = config["mlflow_config"]
    model_name = mlflow_config["registered_model_name"]

    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    runs = mlflow.search_runs(experiment_ids=[1])
    highest_accuracy_score = runs["metrics.r2_score"].sort_values(ascending=True)[
        0]
    highest_run_id = runs[runs["metrics.r2_score"]
                          == highest_accuracy_score]["run_id"][0]

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        if mv["run_id"] == highest_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    model_path = config["prediction"]["final-model"]

    joblib.dump(loaded_model, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)