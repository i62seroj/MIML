import numpy as np

from .hausdorff_distance import HausdorffDistance


class AverageHausdorff(HausdorffDistance):
    """Bidirectional Average Hausdorff distance between two bags.

    The distance is calculated by finding the closest instance in the
    other bag for every instance in each direction. The resulting
    distances are averaged and then normalized to the range ``[0, 1)``.

    A small positive value is returned instead of zero to improve
    numerical stability in algorithms that use the distance as a
    denominator.
    """

    def distance(self, bag1, bag2):
        """Calculate the Average Hausdorff distance between two bags.

        Parameters
        ----------
        bag1 : Bag
            First bag.
        bag2 : Bag
            Second bag.

        Returns
        -------
        float
            Normalized bidirectional Average Hausdorff distance.
        """
        distance_matrix = self._distance_matrix(bag1, bag2)

        # Minimum distance from each instance in bag1 to bag2.
        min_distances_1 = np.min(distance_matrix, axis=1)

        # Minimum distance from each instance in bag2 to bag1.
        min_distances_2 = np.min(distance_matrix, axis=0)

        # Average the minimum distances in both directions.
        distance = (
            np.sum(min_distances_1) + np.sum(min_distances_2)
        ) / (
            len(min_distances_1) + len(min_distances_2)
        )

        # Normalize the distance to avoid unbounded values.
        distance = distance / (1.0 + distance)

        # Avoid returning exactly zero for numerical stability.
        return max(distance, 1e-10)