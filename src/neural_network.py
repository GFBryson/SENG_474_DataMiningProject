from os import path

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# from utils.command_parser import neural_network_commandline_parser as parser
from utils.get_dataset import get_dataset

project_path = path.abspath(path.dirname(__file__))


def main():
    datasets = get_dataset()
    for dataset in datasets:
        X = dataset["X"]
        y = dataset["y"]
        test_percentages = dataset["test_percentage"]
        layer_sizes = dataset["layer_sizes"]

        percentage = test_percentage_test(X, test_percentages, y, 5)
        layer = test_layer_size(X, 0.2, y, layer_sizes)

        features_string_commas = ', '.join(dataset['feature_names'])
        features_string_dash = "-".join([x.split("_")[0] for x in dataset['feature_names']])

        fig, ax = plt.subplots()
        plt.scatter(percentage.keys(), percentage.values())
        plt.ylabel("Accuracy Score")
        plt.xlabel("% of data used to train")
        plt.text(0,1,f"features used: {features_string_commas}", wrap=True,transform=ax.transAxes, fontsize='xx-small')
        output_path = path.join(project_path, "..", "outputs", f"neural_percentage_{dataset['tag']}_{features_string_dash}")
        plt.savefig(output_path)

        plt.clf()

        fig, ax = plt.subplots()
        plt.scatter(layer.keys(), layer.values())
        plt.ylabel("Accuracy Score")
        plt.xlabel("Number of Layers")
        plt.text(0,1,f"features used: {features_string_commas}", wrap=True,transform=ax.transAxes, fontsize='xx-small')
        output_path = path.join(project_path, "..", "outputs", f"neural_layers_{dataset['tag']}_{features_string_dash}")
        plt.savefig(output_path)
        plt.clf()


def test_percentage_test(X, test_percentages, y, layer_size):
    accuracies = {}
    for percentage in test_percentages:
        acc = 0
        for i in range(5):
            acc += get_accuracy(X, percentage, y, "lbfgs", layer_size)
        accuracies[percentage] = acc / 5
    print("test", accuracies)
    return accuracies


def test_layer_size(X, percentage, y, layer_size):
    accuracies = {}
    for size in layer_size:
        acc = 0
        for i in range(5):
            acc += get_accuracy(X, percentage, y, "lbfgs", int(size))
        accuracies[size] = acc / 5
    print("layer", accuracies)
    return accuracies


def get_accuracy(X, test_percentage, y, solver, layer_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage)
    classifier = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=(layer_size, 2), random_state=1,
                               max_iter=1000).fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    acc = metrics.accuracy_score(y_test, prediction)
    # print("Accuracy:", acc)
    return acc


if __name__ == '__main__':
    # program_args = parser().parse_args()
    # main(program_args.dataset)
    main()
    exit()
