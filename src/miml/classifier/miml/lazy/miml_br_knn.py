from enum import Enum

import numpy as np

from .multi_instance_multi_label_knn import MultiInstanceMultiLabelKNN


class ExtensionType(Enum):
    """Extensions available for MIMLBRkNN."""

    NONE = "NONE"
    EXTA = "EXTA"
    EXTB = "EXTB"


class MIMLBRkNN(MultiInstanceMultiLabelKNN):
    """Multi-Instance Multi-Label Binary Relevance kNN classifier.

    Parameters
    ----------
    num_of_neighbours : int, default=10
        Number of nearest neighbours used for prediction.
    metric : HausdorffDistance, optional
        Distance metric used to compare bags.
    extension : ExtensionType, default=ExtensionType.NONE
        Extension applied to the standard binary relevance prediction.
    """

    def __init__(
        self,
        num_of_neighbours=10,
        metric=None,
        extension=ExtensionType.NONE,
    ):
        super().__init__(metric)

        self.k = num_of_neighbours
        self.extension = extension

        self.bags = None
        self.Y = None

    def build_internal(self, training_set):
        """Prepare the training data.

        Parameters
        ----------
        training_set : Dataset
            Dataset used for training.
        """
        self.bags = list(training_set.data.values())

        self.Y = np.array([
            self._extract_labels(bag)
            for bag in self.bags
        ])

    def make_prediction_internal(self, bag):
        """Predict the labels of a bag.

        Parameters
        ----------
        bag : Bag
            Bag to classify.

        Returns
        -------
        numpy.ndarray
            Binary label prediction.
        """
        label_counts, neighbour_labels = (
            self._get_neighbor_label_counts(bag)
        )

        prediction = (
            label_counts >= (self.k / 2)
        ).astype(int)

        return self._apply_extension(
            prediction,
            label_counts,
            neighbour_labels,
        )

    def predict_proba_bag(self, bag):
        """Calculate label probabilities for a bag.

        The probability of a label is calculated as the proportion of
        neighbours containing that label.

        Parameters
        ----------
        bag : Bag
            Bag to classify.

        Returns
        -------
        numpy.ndarray
            Probability for each label.
        """
        label_counts, _ = self._get_neighbor_label_counts(bag)

        return label_counts / self.k

    def _get_neighbor_label_counts(self, bag):
        """Get label frequencies among the nearest neighbours.

        Parameters
        ----------
        bag : Bag
            Bag for which neighbours are searched.

        Returns
        -------
        tuple
            Label counts and neighbour label matrix.
        """
        distances = []

        for index, training_bag in enumerate(self.bags):
            distance = self.metric.distance(
                bag,
                training_bag,
            )

            distances.append((index, distance))

        distances.sort(key=lambda item: item[1])

        neighbour_indices = [
            index
            for index, _ in distances[:self.k]
        ]

        neighbour_labels = self.Y[neighbour_indices]

        label_counts = np.sum(
            neighbour_labels,
            axis=0,
        )

        return label_counts, neighbour_labels

    def _apply_extension(
        self,
        prediction,
        label_counts,
        neighbour_labels,
    ):
        """Apply the selected MIMLBRkNN extension.

        Parameters
        ----------
        prediction : numpy.ndarray
            Initial binary prediction.
        label_counts : numpy.ndarray
            Number of neighbours containing each label.
        neighbour_labels : numpy.ndarray
            Labels of the nearest neighbours.

        Returns
        -------
        numpy.ndarray
            Final binary prediction.
        """
        if self.extension == ExtensionType.NONE:
            return prediction

        if self.extension == ExtensionType.EXTA:
            if np.sum(prediction) == 0:
                max_label = np.argmax(label_counts)
                prediction[max_label] = 1

            return prediction

        if self.extension == ExtensionType.EXTB:
            average_size = int(
                np.round(
                    np.mean(
                        np.sum(
                            neighbour_labels,
                            axis=1,
                        )
                    )
                )
            )

            if average_size == 0:
                return prediction

            top_labels = np.argsort(
                label_counts
            )[::-1][:average_size]

            extended_prediction = np.zeros_like(
                prediction
            )

            extended_prediction[top_labels] = 1

            return extended_prediction

        return prediction

    @staticmethod
    def _extract_labels(bag):
        """Extract the bag-level multilabel representation.

        Parameters
        ----------
        bag : Bag
            Bag from which labels are extracted.

        Returns
        -------
        numpy.ndarray
            Binary label vector.
        """
        labels = bag.get_labels()

        if labels.ndim == 2:
            return labels[0]

        return labels

    def get_extension(self):
        """Return the configured extension."""
        return self.extension

    def set_extension(self, extension):
        """Set the extension used by the classifier.

        Parameters
        ----------
        extension : ExtensionType
            New extension.
        """
        self.extension = extension