import argparse
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def retrieve_params(config):
    with open(config) as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params


def trainTestSplit(config):
    params = retrieve_params(config=config)
    df = pd.read_csv(params["data"]["processed_data"])
    X_train, X_test, y_train, y_test = train_test_split(
                                        df.iloc[:,:-1], df.iloc[:,-1], 
                                        train_size=params["model_training"]["splitting_params"]["split_size"],
                                        random_state=params["model_training"]["splitting_params"]["random_state"])
    
    return X_train, X_test, y_train, y_test

def standardizing(X_train, X_test):
    gre_scaler = StandardScaler()
    cgpa_scaler = StandardScaler()
    X_train["GRE Score"] = gre_scaler.fit_transform(
                                    np.array(X_train["GRE Score"]).reshape(-1,1))
    X_test["GRE Score"] = gre_scaler.transform(
                                    np.array(X_test["GRE Score"]).reshape(-1,1))


    X_train["CGPA"] = cgpa_scaler.fit_transform(
                                    np.array(X_train["CGPA"]).reshape(-1,1))
    X_test["CGPA"] = cgpa_scaler.transform(
                                    np.array(X_test["CGPA"]).reshape(-1,1))

    return X_train, X_test

def train_model(config):
    X_train, X_test, y_train, y_test = train_test_split(config)
    X_train, X_test = standardizing(X_train, X_test)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    trainTestSplit(parsed_args.config)