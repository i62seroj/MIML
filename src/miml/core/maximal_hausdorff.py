import numpy as np

from .hausdorff_distance import HausdorffDistance


class MaximalHausdorff(HausdorffDistance):
    """Directed Hausdorff distance from the first bag to the second.

    For every instance in ``bag1``, the distance to its closest instance
    in ``bag2`` is calculated. The maximum of these minimum distances
    is returned.

    This is a directed distance, meaning that:

    ``distance(bag1, bag2)``

    is not necessarily equal to:

    ``distance(bag2, bag1)``.
    """

    def distance(self, bag1, bag2):
        """Calculate the directed Hausdorff distance.

        Parameters
        ----------
        bag1 : Bag
            Source bag.
        bag2 : Bag
            Target bag.

        Returns
        -------
        float
            Directed Hausdorff distance from ``bag1`` to ``bag2``.
        """
        distance_matrix = self._distance_matrix(bag1, bag2)

        # Find the closest instance in bag2 for each instance in bag1.
        minimum_distances = np.min(distance_matrix, axis=1)

        # The Hausdorff distance is the largest of these minimum distances.
        return np.max(minimum_distances)