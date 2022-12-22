# Possible chance of outlier in LOR and CGPA as seen from box plots
# Toefl score and GRE show a high correlation
# Although GRE score and CGPA show high correlation, based on field specific experience they both play an important role

import argparse
import yaml
import pandas as pd


def retrieve_params(config):
    with open(config) as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params


def calculate_upper_lower_bounds(df, column_name):
    q1 = df[column_name].describe()[4]
    q3 = df[column_name].describe()[6]
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * (iqr)
    lower_limit = q1 - 1.5 * (iqr)
    return upper_limit, lower_limit


def build_feature(config):
    params = retrieve_params(config=config)
    df = pd.read_csv(params["data"]["raw_data"])
    df.columns = [i.strip() for i in df.columns]
    df.drop(["Serial No.", "TOEFL Score"], axis=1, inplace=True)
    lor_upper_limit, lor_lower_limit = calculate_upper_lower_bounds(df, "LOR")
    cgpa_upper_limit, cgpa_lower_limit = calculate_upper_lower_bounds(df, "CGPA")

    lor_drop_indices = list(
        df[(df["LOR"] > lor_upper_limit) | (df["LOR"] < lor_lower_limit)].index
    )
    cgpa_drop_indices = list(
        df[(df["CGPA"] > cgpa_upper_limit) | (df["CGPA"] < cgpa_lower_limit)].index
    )

    indices_to_drop = lor_drop_indices + cgpa_drop_indices
    df.drop(indices_to_drop, inplace=True)

    df.to_csv(params["data"]["processed_data"], index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    build_feature(parsed_args.config)
