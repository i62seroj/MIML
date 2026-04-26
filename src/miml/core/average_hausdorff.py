import numpy as np
from .hausdorff_distance import HausdorffDistance


class AverageHausdorff(HausdorffDistance):
    """
    Average Hausdorff distance (bidirectional)
    """

    def distance(self, bag1, bag2):
        D = self._distance_matrix(bag1, bag2)

        min_X = np.min(D, axis=1)
        min_Y = np.min(D, axis=0)

        d = (np.sum(min_X) + np.sum(min_Y)) / (len(min_X) + len(min_Y))

        # normalización + estabilidad
        d = d / (1 + d)

        return max(d, 1e-10)
    # def distance(self, bag1, bag2):
    #     D = self._distance_matrix(bag1, bag2)

    #     min_X = np.min(D, axis=1)
    #     min_Y = np.min(D, axis=0)

    #     sum_U = np.sum(min_X)
    #     sum_V = np.sum(min_Y)

    #     return (sum_U + sum_V) / (len(min_X) + len(min_Y))