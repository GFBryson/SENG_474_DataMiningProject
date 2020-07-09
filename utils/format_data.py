def configure_data_from_csv(path: str, keys: list , target_names: list, label: str = "label"):
    """Method that takes in a CSV style data string and a list of keys and creates an object with the data split from the labels
    The number of keys must match the number of columns per row and there must be the same number of columns on e4very row of the file
    File will be split according to CSV conventions"""

    data_string = open(path, "r").read()

    data = []
    target = []
    print(keys)
    label_index = keys.index(label)

    lines = data_string.splitlines()
    for index, line in enumerate(lines):
        attributes = line.split(",")
        if len(attributes) != len(keys):
            raise ValueError(
                f"The number of keys does not match the number of columns on line {index}\nKeys:{len(keys)} Columns:{len(attributes)}")

        label_value = attributes[label_index]
        del attributes[label_index]

        target.append(label_value)
        data.append(attributes)

    return {
        "data": data,
        "target": target,
        "target_names": target_names
    }
