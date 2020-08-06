from os import path

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# from utils.command_parser import neural_network_commandline_parser as parser
from utils.get_dataset import get_datasets

project_path = path.abspath(path.dirname(__file__))


def main():
    datasets = get_datasets()
    for dataset in datasets:
        for dataset_configuration in dataset:
            X = dataset_configuration["X"]
            y = dataset_configuration["y"]
            test_percentages = dataset_configuration["test_percentage"]
            layer_sizes = dataset_configuration["layer_sizes"]

            percentage = test_percentage_test(X, test_percentages, y, 20)
            layer = test_layer_size(X, 0.2, y, layer_sizes)

            features_string_commas = ', '.join(dataset_configuration['feature_names'])
            features_string_dash = "-".join([x.split("_")[0] for x in dataset_configuration['feature_names']])

            fig, ax = plt.subplots()
            plt.scatter(percentage.keys(), percentage.values())
            plt.ylabel("Accuracy Score")
            plt.xlabel("% of data used to train")
            plt.text(0,1,f"features used: {features_string_commas}", wrap=True,transform=ax.transAxes, fontsize='xx-small')
            output_path = path.join(project_path, "..", "outputs", dataset_configuration['tag'], f"neural_percentage_{dataset_configuration['tag']}_{features_string_dash}")
            plt.savefig(output_path)

            plt.clf()

            fig, ax = plt.subplots()
            plt.scatter(layer.keys(), layer.values())
            plt.ylabel("Accuracy Score")
            plt.xlabel("Number of Layers")
            plt.text(0,1,f"features used: {features_string_commas}", wrap=True,transform=ax.transAxes, fontsize='xx-small')
            output_path = path.join(project_path, "..", "outputs", dataset_configuration['tag'], f"neural_layers_{dataset_configuration['tag']}_{features_string_dash}")
            plt.savefig(output_path)
            plt.clf()


def test_percentage_test(X, test_percentages, y, layer_size, test_rounds = 5):
    accuracies = {}
    for percentage in test_percentages:
        acc = 0
        for i in range(test_rounds):
            acc += get_accuracy(X, percentage, y, "lbfgs", layer_size)
        accuracies[percentage] = acc / test_rounds
    print("test", accuracies)
    return accuracies


def test_layer_size(X, percentage, y, layer_size, test_rounds = 5):
    accuracies = {}
    for size in layer_size:
        acc = 0
        for i in range(test_rounds):
            acc += get_accuracy(X, percentage, y, "lbfgs", int(size))
        accuracies[size] = acc / test_rounds
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
