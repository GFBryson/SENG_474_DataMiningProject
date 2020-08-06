from os import path

import matplotlib.pyplot as plt

from src.decision_tree import get_accuracy as tree_accuracy
from src.neural_network import get_accuracy as neural_accuracy
from src.random_forest import get_accuracy as forest_accuracy
from utils.get_dataset import get_datasets

project_path = path.abspath(path.dirname(__file__))


def main():
    datasets = get_datasets()

    accuracies = []
    labels = []
    colours = []
    for dataset in datasets:
        for i, dataset_configuration in enumerate(dataset):
            X = dataset_configuration["X"]
            y = dataset_configuration["y"]

            neural_acc = neural_accuracy(X, 0.3, y, "lbfgs", 20)
            tree_acc = tree_accuracy(X, 0.3, y, "gini")
            forest_acc = forest_accuracy(X, 0.3, y, "gini")

            accuracies.append(neural_acc)
            labels.append(f"{dataset_configuration['tag']}_neural")
            colours.append(i)

            accuracies.append(tree_acc)
            labels.append(f"{dataset_configuration['tag']}_tree")
            colours.append(i)

            accuracies.append(forest_acc)
            labels.append(f"{dataset_configuration['tag']}_forest")
            colours.append(i)

        features_string_commas = ', '.join(dataset_configuration['feature_names'])
        features_string_dash = "-".join([x.split("_")[0] for x in dataset_configuration['feature_names']])

        # plt.figure()
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.scatter(accuracies, labels, c=colours, cmap='rainbow', alpha=1, edgecolors='k')
        plt.xlabel("Accuracy Score")
        plt.ylabel("Method")
        # plt.xticks(rotation=90)
        plt.text(0, 1, f"features used: {features_string_commas}", wrap=True, transform=ax.transAxes,
                 fontsize='xx-small')
        output_path = path.join(project_path, "..", "outputs",
                                f"Overall_Comparison_Of_Best_Options_{features_string_dash}")
        plt.savefig(output_path)
        plt.clf()
    pass


if __name__ == '__main__':
    main()
