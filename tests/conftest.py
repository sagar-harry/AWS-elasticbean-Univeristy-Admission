import pytest
import yaml
import json

@pytest.fixture
def retrieve_params(config_path="params.yaml"):
    with open(config_path) as yaml_file:
        params = yaml.safe_load(yaml_file)

    return params

@pytest.fixture
def retrieve_schema(retrieve_params):
    schema_file = retrieve_params["form_data"]["schema_json"]
    with open(schema_file) as json_file:
        schema_dict = json.load(json_file)
    return schema_dict