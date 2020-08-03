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

    overall_effectiveness = []
    for featureName in datasets[0]["feature_names"]:
            overall_effectiveness.append({featureName: 0})
    # for x in range(0,100):
    for dataset in datasets:
        avg_effectiveness = []
        for featureName in datasets[0]["feature_names"]:
            avg_effectiveness.append({featureName: 0})
        # run multiple times to find an average
        for a in range(0, 100):
            X = dataset["X"]
            y = dataset["y"]
            test_percentages = dataset["test_percentage"]
            splitters = dataset["tree_splitters"]

            # Keep the splits the same so test results are reliable
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

            # this is the accuracy when no features have been removed
            full_percentage = test_percentage_test(X_train, y_train, X_test, y_test)

            features = dataset["feature_names"]

            for a in range(0, len(features)):
                feature_dropped = features[a]
                tempX = X.drop(features[a], 1)
                X_train, X_test, y_train, y_test = train_test_split(tempX, y, test_size=0.5, random_state=1)

                # this is the accuracy when a specific feature is removed (positive means accuracy got worse when removed, negative means accuracy got better when removed)
                percentage = test_percentage_test(X_train, y_train, X_test, y_test)

                relative_effectiveness = full_percentage[0.5] - percentage[0.5]
                for item in avg_effectiveness:
                    if feature_dropped in item.keys():
                        item[feature_dropped] += relative_effectiveness
                for item2 in overall_effectiveness:
                    if feature_dropped in item2.keys():
                        item2[feature_dropped] += relative_effectiveness
        objects = []
        scores = []
        axis=[]
        for c in range(0,len(avg_effectiveness)):
            axis.append(c)
            objects.append(features[c])
            scores.append(list(avg_effectiveness[c].values())[0]/100)

        plt.figure(figsize=(9,8))
        features_string_commas = ', '.join(dataset['feature_names'])
        features_string_dash = "-".join([x.split("_")[0] for x in dataset['feature_names']])

        fig, ax = plt.subplots()
        plt.bar(objects, scores)
        plt.ylabel('Relative Accuracy')
        plt.xlabel('Feature')
        plt.xticks(rotation=90)
        plt.title('Effectiveness of Features')
        plt.tight_layout()
        plt.text(0, 1, f"features used: {features_string_commas}", wrap=True, transform=ax.transAxes,
                 fontsize='xx-small')
        output_path = path.join(project_path, "..", "outputs",
                                f"avg_accuracy_features_{dataset['tag']}_{features_string_dash}")
        plt.savefig(output_path)
        plt.clf()

    objects = []
    scores = []
    axis=[]
    for c in range(0,len(avg_effectiveness)):
        axis.append(c)
        objects.append(features[c])
        scores.append(list(overall_effectiveness[c].values())[0]/(100 * len(dataset)))

    plt.figure(figsize=(9,8))
    plt.bar(objects, scores)
    plt.ylabel('Relative Accuracy')
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.title('Effectiveness of Features')
    plt.tight_layout()
    output_path = path.join(project_path, "..", "outputs", f"avg_accuracy_features_overall_{features_string_dash}")
    plt.savefig(output_path)
    plt.clf()
    pass


def test_percentage_test(X_train, y_train, X_test, y_test):
    accuracies = {}
    percentage = 0.5
    acc = 0
    for i in range(5):
        acc += get_accuracy(X_train, y_train, X_test, y_test)
    accuracies[percentage] = acc / 5
    # print("test", accuracies)
    return accuracies


def test_splitter(X, percentage, y, splitters):
    accuracies = {}
    for splitter in splitters:
        acc = 0
        for i in range(5):
            acc += get_accuracy(X, percentage, y, splitter)
        accuracies[splitter] = acc / 5
    # print("test", accuracies)
    return accuracies


def get_accuracy(X_train, y_train, X_test, y_test):
    classifier = DecisionTreeClassifier(criterion="gini").fit(X_train, y_train)

    prediction = classifier.predict(X_test)

    acc = metrics.accuracy_score(y_test, prediction)
    return acc


if __name__ == '__main__':
    # program_args = parser.decision_tree_commandline_parser().parse_args()
    # main(program_args.dataset)
    main()
