import numpy as np

from .multi_instance_multi_label_knn import MultiInstanceMultiLabelKNN


class MIMLDGC(MultiInstanceMultiLabelKNN):
    """Multi-Instance Multi-Label Data Gravitation Classification.

    Parameters
    ----------
    num_of_neighbours : int, default=10
        Number of nearest neighbours.
    metric : HausdorffDistance, optional
        Distance metric used to compare bags.
    extension : bool, default=False
        If ``True``, all neighbours at the same distance as the
        kth neighbour are included.
    """

    def __init__(
        self,
        num_of_neighbours=10,
        metric=None,
        extension=False,
    ):
        super().__init__(metric)

        self.k = num_of_neighbours
        self.extension = extension

        self.bags = None
        self.Y = None

        self.D = None
        self.NGC = None
        self.weights = None
        self.densities = None

        self.weight_max = -np.inf
        self.weight_min = np.inf

    def build_internal(self, training_set):
        """Train the DGC classifier.

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
        """Calculate the DGC model parameters."""
        n_bags = len(self.bags)

        self.D = self.get_distances(self.bags)

        self.NGC = np.zeros(n_bags)
        self.weights = np.zeros(n_bags)
        self.densities = np.zeros(n_bags)

        for index in range(n_bags):
            neighbours = self._get_knn(index)

            self._compute_weight_density(
                index,
                neighbours,
            )

        for index in range(n_bags):
            if self.weight_max != self.weight_min:
                self.weights[index] = (
                    self.weights[index] - self.weight_min
                ) / (
                    self.weight_max - self.weight_min
                )
            else:
                self.weights[index] = 0.0

            self.NGC[index] = (
                self.densities[index]
                ** self.weights[index]
            )

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
        probabilities = self.predict_proba_bag(bag)

        return (probabilities > 0.5).astype(int)

    def predict_proba_bag(self, bag):
        """Calculate label confidence values for a bag.

        Parameters
        ----------
        bag : Bag
            Bag to classify.

        Returns
        -------
        numpy.ndarray
            Confidence value for each label.
        """
        neighbours = self._get_prediction_neighbours(bag)

        gravity = np.zeros(len(neighbours))

        for index, (bag_index, distance) in enumerate(neighbours):
            if distance == 0:
                distance = 1e-10

            gravity[index] = (
                self.NGC[bag_index]
                / (distance ** 2)
            )

        n_labels = self.Y.shape[1]
        confidence = np.zeros(n_labels)

        for label in range(n_labels):
            positive = 0.0
            negative = 0.0

            for index, (bag_index, _) in enumerate(neighbours):
                if self.Y[bag_index][label] == 1:
                    positive += gravity[index]
                else:
                    negative += gravity[index]

            total = positive + negative

            if total > 0:
                confidence[label] = positive / total

        return confidence

    def _get_prediction_neighbours(self, bag):
        """Find neighbours used for prediction.

        Parameters
        ----------
        bag : Bag
            Test bag.

        Returns
        -------
        list[tuple]
            Pairs containing training-bag indices and distances.
        """
        distances = []

        for index, training_bag in enumerate(self.bags):
            distance = self.metric.distance(
                bag,
                training_bag,
            )

            distances.append((index, distance))

        distances.sort(key=lambda item: item[1])

        if not self.extension:
            return distances[:self.k]

        kth_distance = distances[self.k - 1][1]

        return [
            pair
            for pair in distances
            if pair[1] <= kth_distance
        ]

    def _get_knn(self, index):
        """Get the neighbours of a training bag.

        The bag itself is excluded from the neighbour set.

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

        if not self.extension:
            return [
                neighbour_index
                for neighbour_index, _ in distances[:self.k]
            ]

        kth_distance = distances[self.k - 1][1]

        return [
            neighbour_index
            for neighbour_index, distance in distances
            if distance <= kth_distance
        ]

    def _label_distance(self, index_1, index_2):
        """Calculate the distance between two label vectors.

        Parameters
        ----------
        index_1 : int
            First bag index.
        index_2 : int
            Second bag index.

        Returns
        -------
        float
            Fraction of different labels.
        """
        return np.mean(
            self.Y[index_1] != self.Y[index_2]
        )

    def _compute_weight_density(self, index, neighbours):
        """Calculate density and weight for a training bag.

        Parameters
        ----------
        index : int
            Index of the bag.
        neighbours : list[int]
            Indices of neighbouring bags.
        """
        weight = 1.0
        density = 0.0

        p_dis_y = 0.0
        p_dis_f = 0.0
        p_dis_y_dis_f = 0.0

        n_neighbours = len(neighbours)

        for neighbour in neighbours:
            label_distance = self._label_distance(
                index,
                neighbour,
            )

            feature_distance = self.metric.distance(
                self.bags[index],
                self.bags[neighbour],
            )

            if feature_distance == 0:
                continue

            density += (
                (1.0 - label_distance)
                / feature_distance
            )

            p_dis_y += label_distance
            p_dis_f += feature_distance
            p_dis_y_dis_f += (
                label_distance * feature_distance
            )

        density += 1.0

        if n_neighbours > 0:
            p_dis_y /= n_neighbours
            p_dis_f /= n_neighbours
            p_dis_y_dis_f /= n_neighbours

        if p_dis_y == 0 or p_dis_y == 1:
            weight = 0.0
        else:
            weight = (
                (p_dis_y_dis_f * p_dis_f) / p_dis_y
                - (
                    (1.0 - p_dis_y_dis_f)
                    * p_dis_f
                    / (1.0 - p_dis_y)
                )
            )

        self.weight_max = max(
            self.weight_max,
            weight,
        )

        self.weight_min = min(
            self.weight_min,
            weight,
        )

        self.weights[index] = weight
        self.densities[index] = density