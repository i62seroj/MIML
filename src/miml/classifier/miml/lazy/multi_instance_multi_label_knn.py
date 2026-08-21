from abc import ABC, abstractmethod

import numpy as np

from ....core.average_hausdorff import AverageHausdorff


class MultiInstanceMultiLabelKNN(ABC):
    """Base class for Multi-Instance Multi-Label kNN classifiers.

    This class provides the common functionality shared by MIML
    k-nearest-neighbour classifiers, including:

    - Distance metric management.
    - Model training interface.
    - Single-bag prediction.
    - Dataset evaluation.
    - Probability prediction.
    - Pairwise distance calculation.

    Subclasses are responsible for implementing the actual training
    procedure and the prediction logic.
    """

    def __init__(self, metric=None):
        """Initialize the classifier.

        Parameters
        ----------
        metric : HausdorffDistance, optional
            Distance metric used to compare bags. If ``None``,
            :class:`AverageHausdorff` is used.
        """
        self.metric = (
            AverageHausdorff()
            if metric is None
            else metric
        )

        self.trained = False

    def build(self, training_set):
        """Train the classifier.

        Parameters
        ----------
        training_set : Dataset
            Dataset used to train the classifier.
        """
        self.build_internal(training_set)

        self.trained = True

    def predict(self, instance):
        """Predict the labels of a single bag.

        Parameters
        ----------
        instance : Bag
            Bag to classify.

        Returns
        -------
        numpy.ndarray
            Binary label prediction.
        """
        self._check_trained()

        return self.make_prediction_internal(instance)

    def evaluate(self, dataset_test):
        """Predict labels for all bags in a test dataset.

        Parameters
        ----------
        dataset_test : Dataset
            Dataset containing the bags to classify.

        Returns
        -------
        numpy.ndarray
            Binary predictions with shape
            ``(n_bags, n_labels)``.
        """
        self._check_trained()

        test_bags = list(dataset_test.data.values())

        return np.array([
            self.predict(bag)
            for bag in test_bags
        ])

    def predict_proba(self, dataset_test):
        """Predict label probabilities for all bags in a dataset.

        Parameters
        ----------
        dataset_test : Dataset
            Dataset containing the bags to classify.

        Returns
        -------
        numpy.ndarray
            Label probabilities with shape
            ``(n_bags, n_labels)``.
        """
        self._check_trained()

        test_bags = list(dataset_test.data.values())

        return np.array([
            self.predict_proba_bag(bag)
            for bag in test_bags
        ])

    def get_distances(self, bags):
        """Calculate the pairwise distance matrix between bags.

        Only the upper triangular part is calculated explicitly.
        Since the distance matrix is symmetric, the other half is
        filled using the corresponding values.

        Parameters
        ----------
        bags : list
            List of bags.

        Returns
        -------
        numpy.ndarray
            Symmetric pairwise distance matrix.
        """
        n_bags = len(bags)

        distances = np.zeros(
            (n_bags, n_bags),
            dtype=float,
        )

        for i in range(n_bags):
            for j in range(i + 1, n_bags):
                distance = self.metric.distance(
                    bags[i],
                    bags[j],
                )

                distances[i, j] = distance
                distances[j, i] = distance

        return distances

    def _check_trained(self):
        """Check whether the classifier has been trained.

        Raises
        ------
        RuntimeError
            If the classifier has not been trained.
        """
        if not self.trained:
            raise RuntimeError(
                "The classifier is not trained. "
                "Call build() before prediction."
            )

    @abstractmethod
    def build_internal(self, training_set):
        """Train the classifier.

        This method must be implemented by subclasses.

        Parameters
        ----------
        training_set : Dataset
            Dataset used for training.
        """
        raise NotImplementedError

    @abstractmethod
    def make_prediction_internal(self, instance):
        """Predict the labels of a single bag.

        Parameters
        ----------
        instance : Bag
            Bag to classify.

        Returns
        -------
        numpy.ndarray
            Binary label prediction.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba_bag(self, bag):
        """Predict label probabilities for a single bag.

        Parameters
        ----------
        bag : Bag
            Bag to classify.

        Returns
        -------
        numpy.ndarray
            Probability for each label.
        """
        raise NotImplementedError