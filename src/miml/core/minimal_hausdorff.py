import numpy as np
from .hausdorff_distance import HausdorffDistance


class MinimalHausdorff(HausdorffDistance):
    """
    Minimum distance between any pair of instances
    """

    def distance(self, bag1, bag2):
        D = self._distance_matrix(bag1, bag2)

        return np.min(D)