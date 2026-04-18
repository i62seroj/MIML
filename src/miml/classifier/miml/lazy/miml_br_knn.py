import numpy as np
from enum import Enum

from .multi_instance_multi_label_knn import MultiInstanceMultiLabelKNN
from ....core.average_hausdorff import AverageHausdorff

class ExtensionType(Enum):
    NONE = "NONE"
    EXTA = "EXTA"
    EXTB = "EXTB"


class MIMLBRkNN(MultiInstanceMultiLabelKNN):
    """
    Class to represent Multi instance Multi Label Binary Relevance K Nearest Neighbors
    """
    def __init__(self, metric=AverageHausdorff(), num_of_neighbours=10, extension=ExtensionType.NONE):
        """
        Constructor of the class MIMLBRkNN

        Parameters
        ----------
        metric: the distance metric between bags
        num_of_neighbours: number of neighbours
        extension: type of extension (NONE, EXTA, EXTB)
        """
        super().__init__(metric, num_of_neighbours)

        self.k = num_of_neighbours
        self.extension = extension
        self.metric = metric

        self.train_bags = None
        self.train_labels = None  # matriz binaria (n_bags, n_labels)

        self.extension = extension

    def fit(self, bags, labels):
        """
        Training the classifier

        Parameters
        ----------

        bags: list of Bag
        labels: np.array shape (n_bags, n_labels)
        """
        self.train_bags = bags
        self.train_labels = labels


    def predict(self, bag):
        """
        Predict the bag distance with training bag 

        Parameters
        ----------
        bag
        """
        distances = np.array([
            self.metric.distance(bag, train_bag)
            for train_bag in self.train_bags
        ])

        # Order and take k nearest neighbors
        nn_idx = np.argsort(distances)[:self.k]
        nn_labels = self.train_labels[nn_idx]

        # Count by labels
        label_counts = np.sum(nn_labels, axis=0)

        prediction = (label_counts >= (self.k / 2)).astype(int)

        prediction = self._apply_extension(prediction, label_counts, nn_labels)

        return prediction


    def _apply_extension(self, prediction, label_counts, nn_labels):
        """
        Choose the extension

        Parameters
        ----------
        prediction: type of prediction (NONE, EXTA, EXTB)
        label_counts: counts of labels
        nn_labels: nearst neighbor labels
        """
        # NONE: standart prediction
        if self.extension == ExtensionType.NONE:
            return prediction

        # EXTA: if there is not labels, take the label most frequent
        if self.extension == ExtensionType.EXTA:
            if np.sum(prediction) == 0:
                max_label = np.argmax(label_counts)
                prediction[max_label] = 1
            return prediction

        # EXTB: take the average size of neighbor labels 
        if self.extension == ExtensionType.EXTB:
            avg_size = int(np.round(np.mean(np.sum(nn_labels, axis=1))))

            # coger las labels más frecuentes
            top_labels = np.argsort(label_counts)[::-1][:avg_size]

            new_pred = np.zeros_like(prediction)
            new_pred[top_labels] = 1

            return new_pred

        return prediction


    def predict_batch(self, bags):
        return np.array([self.predict(bag) for bag in bags])


    def build_internal(self, training_set):
        """
        Method to train dataset

        Parameters
        ----------
        training_set: dataset to train
        """
        self.train_bags = list(training_set.data.values())

        self.train_labels = np.array([
            bag.get_labels()[0] for bag in self.train_bags
        ])

        self.fit(self.train_bags, self.train_labels)


    def make_prediction_internal(self, instance):
        """
        Method to predict the distance
        """
        return self.predict(instance)


    def get_extension(self):
        return self.extension


    def set_extension(self, extension):
        self.extension = extension