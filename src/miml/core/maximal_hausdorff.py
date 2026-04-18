import numpy as np
from .hausdorff_distance import HausdorffDistance


class MaximalHausdorff(HausdorffDistance):
    """
    Directed Hausdorff distance (X -> Y)
    """

    def distance(self, bag1, bag2):
        D = self._distance_matrix(bag1, bag2)

        min_X = np.min(D, axis=1)

        return np.max(min_X)