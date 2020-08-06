import json
from operator import itemgetter
from os import path

import pandas as pd
from sklearn import datasets


def get_datasets():
    X = []
    y = []
    project_path = path.abspath(path.dirname(__file__))
    json_path = path.join(project_path, "..", "testing_data.json")
    configs = json.loads(open(json_path, "r").read())
    formatted_trials=[]
    for dataset_configuration in configs:
        keys, datasets, test_percentage, feature_names, layer_sizes, tree_splitters, forrest_splitters = itemgetter(
            "keys",
            "datasets",
            "test_percentage",
            "feature_names",
            "layer_sizes",
            "tree_splitters",
            "forrest_splitters")(
            dataset_configuration)

        formatted_datasets=[]
        for dataset in datasets:

            data_path = path.join(project_path, dataset["data_path"])
            frame = pd.read_csv(data_path, header=None, names=keys)

            formatted_dataset = {
                "tag": dataset["tag"],
                "X": frame[feature_names],
                "y": frame[dataset["label"]],
                "test_percentage": test_percentage,
                "feature_names": feature_names,
                "layer_sizes": layer_sizes,
                "tree_splitters": tree_splitters,
                "forrest_splitters": forrest_splitters
            }
            formatted_datasets.append(formatted_dataset)


        formatted_trials.append(formatted_datasets)


    return formatted_trials
