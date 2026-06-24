import os

import pkg_resources

from .bag import Bag
from .instance import Instance
from .miml_dataset import MIMLDataset


def load_dataset(file: str, from_library: bool = False) -> MIMLDataset:
    """
    Function to load a dataset

    Parameters
    ----------
    file : str
        Path of the dataset file
    from_library : bool, default = False
        Boolean value to load datasets from library

    Returns
    ----------
    dataset : MIMLDataset
        Dataset loaded
    """
    if from_library:
        library_datasets = ["toy.arff", "miml_birds.arff", "miml_birds_random_80train.arff",
                            "miml_birds_random_20test.arff", "miml_text_data.arff", "miml_image_data.arff"]
        if file in library_datasets:
            return load_dataset(pkg_resources.resource_filename('miml', 'datasets/'+file))
        else:
            raise Exception("Datasets available from library are: "+str(library_datasets))

    if file[-4:] == ".csv":
        return load_dataset_csv(file)
    elif file[-5:] == ".arff":
        return load_dataset_arff(file)
    else:
        raise Exception("Filetype is not admitted. You should use an .arff or .csv file")


def load_dataset_csv(file: str):
    """
    Function to load a dataset in csv format

    Parameters
    ----------
    file : str
        Path of the dataset file

    Returns
    -------
    dataset : MIMLDataset
        Dataset loaded
    """

    dataset = MIMLDataset()
    csv_file = open(file)
    dataset.set_name(file.split("/")[-1])
    file_name = os.path.basename(file)
    dataset.set_name(os.path.splitext(file_name)[0])

    num_labels = int(csv_file.readline().replace("\n", ""))

    header_line = csv_file.readline().replace("\n", "").split(",")
    features_name = header_line[1:-num_labels]
    dataset.set_features_name(features_name)
    labels_name = header_line[-num_labels:]
    dataset.set_labels_name(labels_name)

    for line in csv_file:

        data = line.split(",")

        key = data[0]

        values = [float(i) for i in data[1:-num_labels]]
        labels = [int(i) for i in data[-num_labels:]]

        instance = Instance(values + labels)

        if key not in dataset.data:
            bag = Bag(key)
            dataset.add_bag(bag)
        dataset.add_instance(key, instance)

    return dataset


def load_dataset_arff(file: str) -> MIMLDataset:
    """
    Function to load a dataset in arff format

    Parameters
    ----------
    file : str
        Path of the dataset file

    Returns
    -------
    dataset : MIMLDataset
        Dataset loaded
    """
    dataset = MIMLDataset()
    arff_file = open(file)
    features_name = []
    labels_name = []
    flag = 0
    for line in arff_file:

        # We check that the string does not contain spaces on the left and that it is not empty.
        line = line.lstrip()

        if not line or line.startswith("%") or line.startswith("@data"):
            continue

        if line.startswith("@"):

            if line.startswith("@relation"):
                dataset.set_name(line[line.find(" ") + 1:])
            elif line.startswith("@attribute bag relational"):
                flag = 1
            elif line.startswith("@end bag"):
                flag = 2
            elif flag == 1:
                features_name.append(line.split(" ")[1])
                dataset.set_features_name(features_name)
            elif flag == 2:
                labels_name.append(line.split(" ")[1])
                dataset.set_labels_name(labels_name)

        else:
            # Remove line break at the end of the string
            line = line.strip("\n")
            line = line.replace(" ", "")

            # We assume that the first element of each instance is the bag identifier
            key = line[0:line.find(",")]
            delimiter = line[line.find(",")+1]

            # Start the bag data when we find the first '"' and end with the second '"'.
            line = line[line.find(delimiter) + 1:]
            values = line[:line.find(delimiter)]
            # Split the values by instances of the stock exchange
            values = values.split("\\n")

            # The rest of the string consists of the following labels
            labels = line[line.find(delimiter) + 2:]
            labels_values = [int(i) for i in labels.split(",")]

            for v in values:
                values_instance = [float(i) for i in v.split(',')]
                instance = Instance(values_instance + labels_values)
                if key not in dataset.data:
                    bag = Bag(key)
                    bag.add_instance(instance)
                    dataset.add_bag(bag)
                else:
                    dataset.add_instance(key, instance)

    return dataset


def arff_to_csv(file: str) -> None:
    """
    Convert MIML Arff to CSV.

    Parameters
    ----------
    file : str
        Filepath of the file to be converted
    """

    arff = open(file)
    csv = open(file[:-5] + ".csv", "w")
    attrib = []
    flag = 0

    for line in arff:
        # We check that the string does not contain blank spaces on the left or that it is empty.
        line = line.lstrip()
        if line == "":
            continue

        if line.startswith("%") or line.startswith("@"):
            if line.startswith("@attribute") and not line.startswith("@attribute bag relational"):
                attrib.append(line.split()[1])

        else:
            if flag == 0:
                csv.write(','.join(attrib) + '\n')
                flag = 1

            # Remove line break from the end of the string
            line = line.strip("\n")
            line = line.replace(" ", "")
            # We assume that the first element of each instance is the identifier of the bag.
            key = line[0:line.find(",")]
            # print("Key: ", key_bag)
            delimiter = line[line.find(",")+1]

            # Start the bag data when we find the first '“‘ and end with the second ’”'.
            line = line[line.find(delimiter) + 1:]
            values = line[:line.find(delimiter)]
            # Split the values by instances of the bag
            values = values.split("\\n")

            # The rest of the string consists of the following labels
            labels = line[line.find(delimiter) + 2:]

            for v in values:
                csv.write(key + "," + v + "," + labels + "\n")

    arff.close()
    csv.close()
