import numpy as np

from .multi_instance_multi_label_knn import MultiInstanceMultiLabelKNN


class MIMLkNN(MultiInstanceMultiLabelKNN):
    """Multi-Instance Multi-Label kNN classifier.

    Parameters
    ----------
    num_references : int, default=1
        Number of references used for each training bag.
    num_citers : int, default=1
        Number of citers considered for each bag.
    metric : HausdorffDistance, optional
        Distance metric used to compare bags.
    """

    def __init__(
        self,
        num_references=1,
        num_citers=1,
        metric=None,
    ):
        super().__init__(metric)

        self.num_references = num_references
        self.num_citers = num_citers

        self.bags = None
        self.Y = None

        self.D = None
        self.ref_matrix = None

        self.phi_matrix = None
        self.t_matrix = None
        self.weights_matrix = None

    def build_internal(self, training_set):
        """Train the MIMLkNN classifier.

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
        """Calculate all parameters required by MIMLkNN."""
        n_bags = len(self.bags)

        if n_bags <= self.num_references:
            self.num_references = max(
                1,
                n_bags - 1,
            )

        self.D = self.get_distances(self.bags)

        n_labels = self.Y.shape[1]

        self.phi_matrix = np.zeros(
            (n_bags, n_labels)
        )

        self.t_matrix = np.zeros(
            (n_bags, n_labels)
        )

        self._calculate_reference_matrix()

        for index in range(n_bags):
            neighbours = self._get_union_neighbours(
                index
            )

            self.phi_matrix[index] = (
                self._calculate_record_label(
                    neighbours
                )
            )

            self.t_matrix[index] = (
                self._get_bag_labels(index)
            )

        self.weights_matrix = (
            self._get_weights_matrix()
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
        prediction, _ = self._predict_with_confidence(
            bag
        )

        return prediction

    def predict_proba_bag(self, bag):
        """Calculate confidence values for a bag.

        Parameters
        ----------
        bag : Bag
            Bag to classify.

        Returns
        -------
        numpy.ndarray
            Confidence values for each label.
        """
        _, confidence = self._predict_with_confidence(
            bag
        )

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
        record = self._calculate_test_record(bag)

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
            weights = self.weights_matrix[:, label]

            decision_value = np.dot(
                weights,
                record,
            )

            is_positive = decision_value > 0.3

            prediction[label] = int(is_positive)

            confidence[label] = (
                1.0 if is_positive else 0.0
            )

        return prediction, confidence

    def _calculate_reference_matrix(self):
        """Build the reference matrix."""
        n_bags = len(self.bags)

        self.ref_matrix = np.zeros(
            (n_bags, n_bags),
            dtype=int,
        )

        for index in range(n_bags):
            references = self._calculate_bag_references(
                index
            )

            for reference in references:
                self.ref_matrix[index, reference] = 1

    def _calculate_bag_references(self, index):
        """Calculate the references of a training bag.

        Parameters
        ----------
        index : int
            Index of the bag.

        Returns
        -------
        list[int]
            Indices of the nearest reference bags.
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
            for neighbour_index, _ in distances[
                :self.num_references
            ]
        ]

    def _get_references(self, index):
        """Get the references of a bag.

        Parameters
        ----------
        index : int
            Bag index.

        Returns
        -------
        list[int]
            Reference indices.
        """
        return [
            neighbour_index
            for neighbour_index in range(len(self.bags))
            if self.ref_matrix[
                index,
                neighbour_index,
            ] == 1
        ]

    def _get_citers(self, index):
        """Get the nearest citers of a bag.

        Parameters
        ----------
        index : int
            Bag index.

        Returns
        -------
        list[int]
            Citer indices.
        """
        citers = []

        for neighbour_index in range(len(self.bags)):
            if self.ref_matrix[
                neighbour_index,
                index,
            ] == 1:
                citers.append(
                    (
                        neighbour_index,
                        self.D[index][neighbour_index],
                    )
                )

        citers.sort(key=lambda item: item[1])

        return [
            neighbour_index
            for neighbour_index, _ in citers[
                :self.num_citers
            ]
        ]

    def _get_union_neighbours(self, index):
        """Get the union of references and citers.

        Parameters
        ----------
        index : int
            Bag index.

        Returns
        -------
        list[int]
            Unique neighbour indices.
        """
        references = self._get_references(index)
        citers = self._get_citers(index)

        return list(
            set(references + citers)
        )

    def _calculate_record_label(self, neighbours):
        """Count labels in a neighbourhood.

        Parameters
        ----------
        neighbours : list[int]
            Neighbour indices.

        Returns
        -------
        numpy.ndarray
            Label-count vector.
        """
        n_labels = self.Y.shape[1]

        counts = np.zeros(n_labels)

        for index in neighbours:
            counts += self.Y[index]

        return counts

    def _get_bag_labels(self, index):
        """Convert binary labels from {0, 1} to {-1, 1}.

        Parameters
        ----------
        index : int
            Bag index.

        Returns
        -------
        numpy.ndarray
            Transformed label vector.
        """
        return np.where(
            self.Y[index] == 1,
            1.0,
            -1.0,
        )

    def _get_weights_matrix(self):
        """Calculate the regression weights.

        Returns
        -------
        numpy.ndarray
            Weight matrix.
        """
        weights, _, _, _ = np.linalg.lstsq(
            self.phi_matrix,
            self.t_matrix,
            rcond=None,
        )

        return weights

    def _calculate_test_record(self, bag):
        """Calculate the neighbourhood label vector of a test bag.

        Parameters
        ----------
        bag : Bag
            Test bag.

        Returns
        -------
        numpy.ndarray
            Label-count vector.
        """
        n_bags = len(self.bags)

        distances = np.zeros(n_bags)

        for index in range(n_bags):
            distances[index] = self.metric.distance(
                bag,
                self.bags[index],
            )

        references = np.argsort(
            distances
        )[:self.num_references]

        citers = []

        for index in range(n_bags):
            train_references = self._get_references(
                index
            )

            if not train_references:
                continue

            worst_reference_distance = max(
                self.D[index][reference]
                for reference in train_references
            )

            if distances[index] < worst_reference_distance:
                citers.append(index)

        citers.sort(
            key=lambda index: distances[index]
        )

        citers = citers[:self.num_citers]

        neighbours = list(
            set(
                list(references) + citers
            )
        )

        return self._calculate_record_label(
            neighbours
        )