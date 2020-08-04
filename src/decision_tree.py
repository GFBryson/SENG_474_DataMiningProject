from os import path

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

        plt.ylabel("Accuracy Score")

        percentage = test_percentage_test(X, test_percentages, y, "gini")
        splitters = test_splitter(X, 0.2, y, splitters)

        features_string_commas = ', '.join(dataset['feature_names'])
        features_string_dash = "-".join([x.split("_")[0] for x in dataset['feature_names']])

        figure, ax = plt.subplot()
        plt.scatter(percentage.keys(), percentage.values(), marker='o')
        plt.ylabel("Accuracy Score")
        plt.xlabel("% of data used to train")
        plt.text(0,1,f"features used: {features_string_commas}", wrap=True,transform=ax.transAxes, fontsize='xx-small')
        output_path = path.join(project_path, "..", f"./outputs/tree_percentage_{dataset['tag']}_{features_string_dash}")
        plt.savefig(output_path)
        plt.clf()

        figure, ax = plt.subplot()
        plt.scatter(splitters.keys(), splitters.values(), marker='o')
        plt.ylabel("Accuracy Score")
        plt.xlabel("tree splitting protocol")
        plt.text(0,1,f"features used: {features_string_commas}", wrap=True,transform=ax.transAxes, fontsize='xx-small')
        output_path = path.join(project_path, "..", f"./outputs/tree_splitters_{dataset['tag']}_{features_string_dash}")
        plt.savefig(output_path)
        plt.clf()

    pass


def test_percentage_test(X, test_percentages, y, splitter, test_rounds = 5):
    accuracies = {}
    for percentage in test_percentages:
        acc = 0
        for i in range(test_rounds):
            acc += get_accuracy(X, percentage, y, splitter)
        accuracies[percentage] = acc / test_rounds
    print("test", accuracies)
    return accuracies


def test_splitter(X, percentage, y, splitters, test_rounds = 5):
    accuracies = {}
    for splitter in splitters:
        acc = 0
        for i in range(test_rounds):
            acc += get_accuracy(X, percentage, y, splitter)
        accuracies[splitter] = acc / 5
    print("test", accuracies)
    return accuracies


def get_accuracy(X, test_percentage, y, splitter):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=1)

    classifier = DecisionTreeClassifier(criterion=splitter).fit(X_train, y_train)

    prediction = classifier.predict(X_test)

    acc = metrics.accuracy_score(y_test, prediction)
    return acc


if __name__ == '__main__':
    # program_args = parser.decision_tree_commandline_parser().parse_args()
    # main(program_args.dataset)
    main()
