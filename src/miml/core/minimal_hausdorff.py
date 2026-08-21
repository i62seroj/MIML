import numpy as np

from .hausdorff_distance import HausdorffDistance


class MinimalHausdorff(HausdorffDistance):
    """Minimum distance between any pair of instances in two bags.

    The distance is defined as the minimum pairwise distance between
    any instance from ``bag1`` and any instance from ``bag2``.
    """

    def distance(self, bag1, bag2):
        """Calculate the minimum distance between two bags.

        Parameters
        ----------
        bag1 : Bag
            First bag.
        bag2 : Bag
            Second bag.

        Returns
        -------
        float
            Minimum distance between any pair of instances from the
            two bags.
        """
        distance_matrix = self._distance_matrix(bag1, bag2)

        return np.min(distance_matrix)