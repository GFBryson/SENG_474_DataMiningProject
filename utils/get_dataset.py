import json
from operator import itemgetter
from os import path

import pandas as pd
from sklearn import datasets


def get_dataset(dataset: str):
    X = []
    y = []
    project_path = path.abspath(path.dirname(__file__))
    json_path = path.join(project_path, "..", "testing_data.json")
    config = json.loads(open(json_path, "r").read())
    keys, data_path, label, test_percentage, target_names, feature_names, layer_sizes, tree_splitters, forrest_splitters = itemgetter(
        "keys",
        "data_path",
        "label",
        "test_percentage",
        "target_names",
        "feature_names",
        "layer_sizes",
        "tree_splitters",
        "forrest_splitters")(
        config)

    if dataset == "heart":
        data_path = path.join(project_path, data_path)
        frame = pd.read_csv(data_path, header=None, names=keys)

        X = frame[feature_names]
        y = frame[label]

    elif dataset == "flower":
        iris = datasets.load_iris()

        data = pd.DataFrame({
            'sepal length': iris.data[:, 0],
            'sepal width': iris.data[:, 1],
            'petal length': iris.data[:, 2],
            'petal width': iris.data[:, 3],
            'species': iris.target
        })

        X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
        y = data['species']

        target_names = iris.target_names
        feature_names = iris.feature_names
    elif dataset == "cancer":
        cancer = datasets.load_breast_cancer()
        frame_setup = {'status':cancer.target}
        for index, key in enumerate(cancer.feature_names):
            frame_setup[key] = cancer.data[:,index]

        data = pd.DataFrame(frame_setup)

        X = data[cancer.feature_names]
        y = data['status']

        # data = pd.DataFrame({
        #     'sepal length': cancer.data[:, 0],
        #     'sepal width': cancer.data[:, 1],
        #     'petal length': cancer.data[:, 2],
        #     'petal width': cancer.data[:, 3],
        #     'species': cancer.target
        # })
    return {
        "X": X,
        "y": y,
        "test_percentage": test_percentage,
        "target_names": target_names,
        "feature_names": feature_names,
        "layer_sizes": layer_sizes,
        "tree_splitters": tree_splitters,
        "forrest_splitters": forrest_splitters
    }
