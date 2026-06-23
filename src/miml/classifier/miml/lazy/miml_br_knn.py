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
    def __init__(self, num_of_neighbours=10, metric=None, extension=ExtensionType.NONE):
        """
        Constructor of the class MIMLBRkNN

        Parameters
        ----------
        metric: the distance metric between bags
        num_of_neighbours: number of neighbours
        extension: type of extension (NONE, EXTA, EXTB)
        """
        super().__init__(metric, num_of_neighbours)


        if metric == None:
            metric = AverageHausdorff()
        
        self.k = num_of_neighbours
        self.extension = extension
        self.metric = metric

        self.bags = None
        self.Y = None  # matriz binaria (n_bags, n_labels)

        self.extension = extension

    def fit(self, bags, labels):
        """
        Training the classifier

        Parameters
        ----------

        bags: list of Bag
        labels: np.array shape (n_bags, n_labels)
        """
        # n = len(bags)

        # self.D = self.get_distances(bags)


    def make_prediction_internal(self, bag):
        """
        Predict the bag distance with training bag 

        Parameters
        ----------
        bag
        """
        label_counts, nn_labels = self._get_neighbor_label_counts(bag)

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

            if avg_size == 0:
                return prediction
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
        self.bags = list(training_set.data.values())

        def extract_labels(bag):
                labels = bag.get_labels()
                if len(labels.shape) == 2:
                    return labels[0]
                return labels
        
        self.Y = np.array([extract_labels(b) for b in self.bags])

        self.fit(self.bags, self.Y)

        self.trained = True


    # def make_prediction_internal(self, instance):
    #     """
    #     Method to predict the distance
    #     """
    #     return self.predict(instance)


    def get_extension(self):
        return self.extension


    def set_extension(self, extension):
        self.extension = extension

    def _get_neighbor_label_counts(self, bag):
        distances = []

        for i in range(len(self.bags)):
            d = self.metric.distance(bag, self.bags[i])
            distances.append((i, d))

        distances.sort(key=lambda x: x[1])

        nn_idx = [idx for idx, _ in distances[:self.k]]

        nn_labels = self.Y[nn_idx]

        label_counts = np.sum(nn_labels, axis=0)

        return label_counts, nn_labels

    def predict_proba_bag(self, bag):
        label_counts, _ = self._get_neighbor_label_counts(bag)
        return label_counts / self.k

    def predict_proba(self, dataset_test):
        if not self.trained:
            raise Exception(
                "The classifier is not trained. You need to call fit before predict anything"
            )

        test_bags = list(dataset_test.data.values())

        return np.array([
            self.predict_proba_bag(bag)
            for bag in test_bags
        ])