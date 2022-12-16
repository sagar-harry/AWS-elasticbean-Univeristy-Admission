import yaml
import argparse
import shutil

def retrieve_params(config):
    with open(config) as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params

def load_data(config):
    params = retrieve_params(config)
    shutil.copy(params["data"]["input_data"], params["data"]["raw_data"])
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_data(parsed_args.config)
    