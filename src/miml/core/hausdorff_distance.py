from abc import ABC, abstractmethod

from scipy.spatial.distance import cdist


class HausdorffDistance(ABC):
    """Base class for Hausdorff-based distances between MIML bags.

    This class provides the common functionality required by distance
    measures based on the Hausdorff distance.

    Subclasses must implement the :meth:`distance` method.
    """

    def _bags_to_arrays(self, bag1, bag2):
        """Extract feature matrices from two bags.

        Parameters
        ----------
        bag1 : Bag
            First bag.
        bag2 : Bag
            Second bag.

        Returns
        -------
        tuple
            Feature matrices corresponding to ``bag1`` and ``bag2``.
        """
        features_1 = bag1.get_features()
        features_2 = bag2.get_features()

        return features_1, features_2

    def _distance_matrix(self, bag1, bag2):
        """Calculate the pairwise Euclidean distance matrix.

        Parameters
        ----------
        bag1 : Bag
            First bag.
        bag2 : Bag
            Second bag.

        Returns
        -------
        numpy.ndarray
            Pairwise Euclidean distance matrix.
        """
        features_1, features_2 = self._bags_to_arrays(
            bag1,
            bag2,
        )

        return cdist(
            features_1,
            features_2,
            metric="euclidean",
        )

    @abstractmethod
    def distance(self, bag1, bag2):
        """Calculate the distance between two bags.

        Parameters
        ----------
        bag1 : Bag
            First bag.
        bag2 : Bag
            Second bag.

        Returns
        -------
        float
            Distance between the two bags.
        """
        raise NotImplementedError