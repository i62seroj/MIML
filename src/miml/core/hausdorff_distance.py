import numpy as np
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod


class HausdorffDistance(ABC):
    """
    Base class for Hausdorff-based distances between two MIML bags.
    """

    def _bags_to_arrays(self, bag1, bag2):
        X = bag1.get_features()
        Y = bag2.get_features()
        return X, Y

    def _distance_matrix(self, bag1, bag2):
        """
        Compute pairwise distance matrix.
        """
        X, Y = self._bags_to_arrays(bag1, bag2)
        return cdist(X, Y, metric="euclidean")

    @abstractmethod
    def distance(self, bag1, bag2):
        pass