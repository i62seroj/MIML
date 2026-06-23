from abc import ABC
import numpy as np


class MultiInstanceMultiLabelKNN(ABC):
    """
    Abstract class for MIML kNN classifiers
    """
    def __init__(self, metric=None, num_of_neighbours=10):
        """
        Constructor

        Parameters
        ----------
        metric: the distance metric between bags
        num_of_neighbours: number of nearest neighbours
        """
        self.metric = metric
        self.num_of_neighbours = num_of_neighbours
        self.classifier = None
        self.trained = False

    def build(self, training_set):
        self.build_internal(training_set)

    def predict(self, instance):
        return self.make_prediction_internal(instance)

    def build_internal(self, training_set):
        raise NotImplementedError

    def make_prediction_internal(self, instance):
        raise NotImplementedError
    
    def get_distances(self, bags):

        n = len(bags)
        self.D = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                d = self.metric.distance(bags[i], bags[j])
                self.D[i][j] = d
                self.D[j][i] = d
        return self.D

    def evaluate(self, dataset_test):
        if not self.trained:
            raise Exception(
                "The classifier is not trained. You need to call fit before predict anything"
            )

        test_bags = list(dataset_test.data.values())

        return np.array([
            self.predict(bag)
            for bag in test_bags
        ])

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