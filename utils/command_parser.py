import argparse
import json
import os

def parse_json(parser, path):
    if not os.path.exists(path):
        parser.error(f"The file {path} does not exist!")
    else:
        with open(path, "r") as json_file:
            return json.load(json_file)

def decision_tree_commandline_parser():
    parser = argparse.ArgumentParser(description='Generate aggregation airflow graphs and tasks')
    parser.add_argument("--dataset",
                        dest="dataset",
                        required=True,
                        help="heart = use cleaned_processed.cleveland.data dataset\nflower = use isris dataset",
                        metavar="str"
                        )

    return parser

def random_forrest_commandline_parser():
    parser = argparse.ArgumentParser(description='Generate aggregation airflow graphs and tasks')
    parser.add_argument("--dataset",
                        dest="dataset",
                        required=True,
                        help="heart = use cleaned_processed.cleveland.data dataset\nflower = use isris dataset",
                        metavar="str"
                        )

    return parser

def neural_network_commandline_parser():
    parser = argparse.ArgumentParser(description='Generate aggregation airflow graphs and tasks')
    parser.add_argument("--dataset",
                        dest="dataset",
                        required=True,
                        help="heart = use cleaned_processed.cleveland.data dataset\nflower = use isris dataset",
                        metavar="str"
                        )

    return parser