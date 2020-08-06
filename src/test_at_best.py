from os import path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.decision_tree import get_accuracy as tree_accuracy
from src.neural_network import get_accuracy as neural_accuracy
from src.random_forest import get_accuracy as forest_accuracy
from utils.get_dataset import get_datasets

project_path = path.abspath(path.dirname(__file__))
avail_markers=['o','P','v','*']

def main():
    datasets = get_datasets()

    for dataset in datasets:
        accuracies = []
        method_labels = []
        labels = []
        colours = []
        markers = []
        for i, dataset_configuration in enumerate(dataset):
            X = dataset_configuration["X"]
            y = dataset_configuration["y"]

            neural_acc = neural_accuracy(X, 0.3, y, "lbfgs", 20)
            tree_acc = tree_accuracy(X, 0.3, y, "gini")
            forest_acc = forest_accuracy(X, 0.3, y, "gini")

            accuracies.append(neural_acc)
            method_labels.append(f"{dataset_configuration['tag']}_neural")
            labels.append("neural")
            markers.append(avail_markers[i])
            colours.append("r")

            accuracies.append(tree_acc)
            method_labels.append(f"{dataset_configuration['tag']}_tree")
            labels.append("tree")
            markers.append(avail_markers[i])
            colours.append("g")

            accuracies.append(forest_acc)
            method_labels.append(f"{dataset_configuration['tag']}_forest")
            labels.append("forest")
            markers.append(avail_markers[i])
            colours.append("b")

        features_string_commas = ', '.join(dataset_configuration['feature_names'])
        features_string_dash = "-".join([x.split("_")[0] for x in dataset_configuration['feature_names']])

        # plt.figure()
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, a in enumerate(accuracies):
            ax.scatter(accuracies[i], method_labels[i],label=labels[i], c=colours[i], alpha=1, edgecolors=None, marker=markers[i])
        plt.xlabel("Accuracy Score")
        plt.ylabel("Method")
        # ax.legend()
        # plt.xticks(rotation=90)

        N = mpatches.Patch(color='r', label='neural')
        T = mpatches.Patch(color='g', label='tree')
        F = mpatches.Patch(color='b', label='forest')
        ax.legend(handles=[N,T,F],bbox_to_anchor=(1.135, 1))

        plt.text(0, 1, f"features used: {features_string_commas}", wrap=True, transform=ax.transAxes,
                 fontsize='xx-small')
        output_path = path.join(project_path, "..", "outputs",
                                f"Best_Options_{features_string_dash}")
        plt.savefig(output_path)
        plt.clf()
    pass


if __name__ == '__main__':
    main()
