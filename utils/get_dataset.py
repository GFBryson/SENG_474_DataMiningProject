import json
from operator import itemgetter
from os import path

import pandas as pd
from sklearn import datasets


def get_dataset():
    X = []
    y = []
    project_path = path.abspath(path.dirname(__file__))
    json_path = path.join(project_path, "..", "testing_data.json")
    config = json.loads(open(json_path, "r").read())
    keys, data_path, WK1_label, WK2_label, WK3_label, WK4_label, test_percentage, target_names, feature_names, layer_sizes, tree_splitters, forrest_splitters = itemgetter(
        "keys",
        "data_path",
        "WK1_label",
        "WK2_label",
        "WK3_label",
        "WK4_label",
        "test_percentage",
        "target_names",
        "feature_names",
        "layer_sizes",
        "tree_splitters",
        "forrest_splitters")(
        config)

    data_path = path.join(project_path, data_path)
    frame = pd.read_csv(data_path, header=None, names=keys)

    X = frame[feature_names]

    return [
        {
            "tag": "WK1",
            "X": X,
            "y": frame[WK1_label],
            "test_percentage": test_percentage,
            "target_names": target_names,
            "feature_names": feature_names,
            "layer_sizes": layer_sizes,
            "tree_splitters": tree_splitters,
            "forrest_splitters": forrest_splitters
        },{
            "tag": "WK2",
            "X": X,
            "y": frame[WK2_label],
            "test_percentage": test_percentage,
            "target_names": target_names,
            "feature_names": feature_names,
            "layer_sizes": layer_sizes,
            "tree_splitters": tree_splitters,
            "forrest_splitters": forrest_splitters
        },{
            "tag": "WK3",
            "X": X,
            "y": frame[WK3_label],
            "test_percentage": test_percentage,
            "target_names": target_names,
            "feature_names": feature_names,
            "layer_sizes": layer_sizes,
            "tree_splitters": tree_splitters,
            "forrest_splitters": forrest_splitters
        },{
            "tag": "WK4",
            "X": X,
            "y": frame[WK4_label],
            "test_percentage": test_percentage,
            "target_names": target_names,
            "feature_names": feature_names,
            "layer_sizes": layer_sizes,
            "tree_splitters": tree_splitters,
            "forrest_splitters": forrest_splitters
        }
    ]
