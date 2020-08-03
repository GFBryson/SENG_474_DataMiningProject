from os import path

import matplotlib.pyplot as plt
import pandas as pd
from numpy import array
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# from utils import command_parser as parser
from utils.get_dataset import get_dataset

project_path = path.abspath(path.dirname(__file__))


def main():
    datasets = get_dataset()

    for dataset in datasets:
        X = dataset["X"]
        y = dataset["y"]
        test_percentages = dataset["test_percentage"]
        splitters = dataset["tree_splitters"]

        percentage = test_percentage_test(X, test_percentages, y, "gini")
        splitters = test_splitter(X, 0.2, y, splitters)

        features_string_commas = ', '.join(dataset['feature_names'])
        features_string_dash = "-".join([x.split("_")[0] for x in dataset['feature_names']])

        figure, ax = plt.subplot()
        plt.scatter(percentage.keys(), percentage.values())
        plt.ylabel("Accuracy Score")
        plt.xlabel("% of data used to train")
        plt.text(0,1,f"features used: {features_string_commas}", wrap=True,transform=ax.transAxes, fontsize='xx-small')
        output_path = path.join(project_path, "..", f"./outputs/forrest_percentage_{dataset['tag']}_{features_string_dash}")
        plt.savefig(output_path)
        plt.clf()

        figure, ax = plt.subplot()
        plt.scatter(splitters.keys(), splitters.values())
        plt.ylabel("Accuracy Score")
        plt.xlabel("tree splitting protocol")
        plt.text(0,1,f"features used: {features_string_commas}", wrap=True,transform=ax.transAxes, fontsize='xx-small')
        output_path = path.join(project_path, "..", f"./outputs/forrest_splitters_{dataset['tag']}_{features_string_dash}")
        plt.savefig(output_path)
        plt.clf()


def test_percentage_test(X, test_percentages, y, splitter):
    accuracies = {}
    for percentage in test_percentages:
        acc = 0
        for i in range(5):
            acc += get_accuracy(X, percentage, y, splitter)
        accuracies[percentage] = acc / 5
    print("test", accuracies)
    return accuracies


def test_splitter(X, percentage, y, splitters):
    accuracies = {}
    for splitter in splitters:
        acc = 0
        for i in range(5):
            acc += get_accuracy(X, percentage, y, splitter)
        accuracies[splitter] = acc / 5
    print("splitter", accuracies)
    return accuracies


def get_accuracy(X, percentage, y, splitter):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentage)

    classifier = RandomForestClassifier(criterion=splitter, n_estimators=100).fit(X_train, y_train)

    prediction = classifier.predict(X_test)

    return metrics.accuracy_score(y_test, prediction)


def get_data_frame(data, features, label, target):
    data = array(data)
    target = array(target)

    frame = {}
    for index, key in enumerate(features):
        frame[key] = data[:, index]
    frame[label] = target
    data_frame = pd.DataFrame(frame)

    return data_frame


if __name__ == '__main__':
    # program_args = parser.random_forrest_commandline_parser().parse_args()
    # main(program_args.dataset)
    main()
    exit()
