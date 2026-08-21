import numpy as np

from .multi_instance_multi_label_knn import MultiInstanceMultiLabelKNN


class MIMLMAPkNN(MultiInstanceMultiLabelKNN):
    """Multi-Instance Multi-Label MAP kNN classifier.

    Parameters
    ----------
    num_of_neighbours : int, default=10
        Number of neighbours used by the classifier.
    metric : HausdorffDistance, optional
        Distance metric used to compare bags.
    smooth : float, default=1.0
        Laplace smoothing parameter.
    """

    def __init__(
        self,
        num_of_neighbours=10,
        metric=None,
        smooth=1.0,
    ):
        super().__init__(metric)

        self.k = num_of_neighbours
        self.smooth = smooth

        self.bags = None
        self.Y = None
        self.D = None

        self.prior = None
        self.prior_n = None

        self.cond = None
        self.cond_n = None

    def build_internal(self, training_set):
        """Train the MAP kNN classifier.

        Parameters
        ----------
        training_set : Dataset
            Dataset used for training.
        """
        self.bags = list(training_set.data.values())

        self.Y = np.array([
            np.max(
                bag.get_labels(),
                axis=0,
            )
            for bag in self.bags
        ])

        self._fit()

    def _fit(self):
        """Calculate all MAP kNN model parameters."""
        n_bags = len(self.bags)
        n_labels = self.Y.shape[1]

        self.D = self.get_distances(self.bags)

        self.prior = np.zeros(n_labels)
        self.prior_n = np.zeros(n_labels)

        self.cond = np.zeros(
            (n_labels, self.k + 1)
        )

        self.cond_n = np.zeros(
            (n_labels, self.k + 1)
        )

        self._compute_prior(
            n_bags,
            n_labels,
        )

        self._compute_conditional(
            n_bags,
            n_labels,
        )

    def _compute_prior(self, n_bags, n_labels):
        """Calculate prior label probabilities.

        Parameters
        ----------
        n_bags : int
            Number of training bags.
        n_labels : int
            Number of labels.
        """
        for label in range(n_labels):
            positives = np.sum(
                self.Y[:, label]
            )

            self.prior[label] = (
                self.smooth + positives
            ) / (
                self.smooth * 2 + n_bags
            )

            self.prior_n[label] = (
                1.0 - self.prior[label]
            )

    def _compute_conditional(self, n_bags, n_labels):
        """Calculate conditional probabilities.

        Parameters
        ----------
        n_bags : int
            Number of training bags.
        n_labels : int
            Number of labels.
        """
        temp_c = np.zeros(
            (n_labels, self.k + 1),
            dtype=int,
        )

        temp_nc = np.zeros(
            (n_labels, self.k + 1),
            dtype=int,
        )

        for index in range(n_bags):
            neighbours = self._get_neighbors(index)

            for label in range(n_labels):
                positive_neighbours = sum(
                    self.Y[neighbour][label] == 1
                    for neighbour in neighbours
                )

                if self.Y[index][label] == 1:
                    temp_c[
                        label,
                        positive_neighbours,
                    ] += 1
                else:
                    temp_nc[
                        label,
                        positive_neighbours,
                    ] += 1

        for label in range(n_labels):
            total_positive = np.sum(
                temp_c[label]
            )

            total_negative = np.sum(
                temp_nc[label]
            )

            for count in range(self.k + 1):
                self.cond[label][count] = (
                    self.smooth
                    + temp_c[label][count]
                ) / (
                    self.smooth * (self.k + 1)
                    + total_positive
                )

                self.cond_n[label][count] = (
                    self.smooth
                    + temp_nc[label][count]
                ) / (
                    self.smooth * (self.k + 1)
                    + total_negative
                )

    def _get_neighbors(self, index):
        """Get the k nearest training neighbours.

        Parameters
        ----------
        index : int
            Index of the training bag.

        Returns
        -------
        list[int]
            Indices of the nearest neighbours.
        """
        distances = []

        for neighbour_index in range(len(self.bags)):
            if neighbour_index == index:
                continue

            distances.append(
                (
                    neighbour_index,
                    self.D[index][neighbour_index],
                )
            )

        distances.sort(key=lambda item: item[1])

        return [
            neighbour_index
            for neighbour_index, _ in distances[:self.k]
        ]

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
        prediction, _ = self._predict_with_confidence(bag)

        return prediction

    def predict_proba_bag(self, bag):
        """Calculate label probabilities for a bag.

        Parameters
        ----------
        bag : Bag
            Bag to classify.

        Returns
        -------
        numpy.ndarray
            Probability for each label.
        """
        _, confidence = self._predict_with_confidence(bag)

        return confidence

    def _predict_with_confidence(self, bag):
        """Calculate predictions and confidence values.

        Parameters
        ----------
        bag : Bag
            Bag to classify.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Binary predictions and confidence values.
        """
        distances = []

        for index, training_bag in enumerate(self.bags):
            distance = self.metric.distance(
                bag,
                training_bag,
            )

            distances.append((index, distance))

        distances.sort(key=lambda item: item[1])

        neighbours = distances[:self.k]

        n_labels = self.Y.shape[1]

        prediction = np.zeros(
            n_labels,
            dtype=int,
        )

        confidence = np.zeros(
            n_labels,
            dtype=float,
        )

        for label in range(n_labels):
            positive_neighbours = sum(
                self.Y[index][label] == 1
                for index, _ in neighbours
            )

            p_positive = (
                self.prior[label]
                * self.cond[
                    label,
                    positive_neighbours,
                ]
            )

            p_negative = (
                self.prior_n[label]
                * self.cond_n[
                    label,
                    positive_neighbours,
                ]
            )

            if p_positive > p_negative:
                prediction[label] = 1

            total_probability = (
                p_positive + p_negative
            )

            if total_probability > 0:
                confidence[label] = (
                    p_positive
                    / total_probability
                )

        return prediction, confidence