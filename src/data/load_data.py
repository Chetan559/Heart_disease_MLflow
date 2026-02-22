import pandas as pd
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(config):
    train = pd.read_csv(config["paths"]["train"])
    test = pd.read_csv(config["paths"]["test"])

    test_ids = test["id"]
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)

    return train, test, test_ids