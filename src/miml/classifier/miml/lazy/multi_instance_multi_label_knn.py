from abc import ABC


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

    def build(self, training_set):
        self.build_internal(training_set)

    def predict(self, instance):
        return self.make_prediction_internal(instance)

    def build_internal(self, training_set):
        raise NotImplementedError

    def make_prediction_internal(self, instance):
        raise NotImplementedError