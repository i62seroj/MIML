import numpy as np

from .multi_instance_multi_label_knn import MultiInstanceMultiLabelKNN
from ....core.average_hausdorff import AverageHausdorff


class MIMLkNN(MultiInstanceMultiLabelKNN):

    def __init__(
        self,
        num_references=1,
        num_citers=1,
        metric=None
    ):
        """
        Constructor

        Parameters
        ----------
        num_references : number of references (nearest neighbours)
        num_citers : number of citers
        metric : distance metric
        """

        super().__init__(metric)

        if metric is None:
            metric = AverageHausdorff()

        self.metric = metric

        self.num_references = num_references
        self.num_citers = num_citers

        self.bags = None
        self.Y = None

        self.ref_matrix = None

        self.phi_matrix = None
        self.t_matrix = None

        self.weights_matrix = None

    def build_internal(self, training_set):
        """
        Train classifier

        Parameters
        ----------
        training_set
        """

        self.bags = list(training_set.data.values())

        self.Y = np.array([
            bag.get_labels()[0]
            for bag in self.bags
        ])

        self.fit(self.bags)

    def fit(self, bags):
        """
        Train classifier

        Parameters
        ----------
        bags : list[Bag]
        """

        n = len(bags)

        if n <= self.num_references:
            self.num_references = n - 1

        self.D = self.get_distances(bags)

        n_labels = self.Y.shape[1]

        self.phi_matrix = np.zeros((n, n_labels))

        self.t_matrix = np.zeros((n, n_labels))

        self._calculate_reference_matrix()

        for i in range(n):

            neighbours = self._get_union_neighbours(i)

            self.phi_matrix[i] = self._calculate_record_label(neighbours)

            self.t_matrix[i] = self._get_bag_labels(i)

        self.weights_matrix = self._get_weights_matrix()

    def make_prediction_internal(self, bag):
        """
        Predict labels for a bag

        Parameters
        ----------
        bag : Bag

        Returns
        -------
        bipartition, confidence
        """

        record = self._calculate_test_record(bag)

        n_labels = self.Y.shape[1]

        bipartition = np.zeros(n_labels, dtype=int)

        confidence = np.zeros(n_labels)

        for label in range(n_labels):

            weights = self.weights_matrix[:, label]

            decision_value = np.dot(weights, record)

            prediction = decision_value > 0.3

            bipartition[label] = int(prediction)
            confidence[label] = 1.0 if prediction else 0.0

        return bipartition, confidence

    def _calculate_reference_matrix(self):
        """
        Build reference matrix
        """

        n = len(self.bags)

        self.ref_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):

            refs = self._calculate_bag_references(i)

            for r in refs:
                self.ref_matrix[i][r] = 1

    def _calculate_bag_references(self, index):
        """
        Calculate references of a bag

        Parameters
        ----------
        index : int

        Returns
        -------
        list[int]
        """

        distances = []

        for j in range(len(self.bags)):

            if j == index:
                continue

            distances.append((j, self.D[index][j]))

        distances.sort(key=lambda x: x[1])

        return [
            idx
            for idx, _
            in distances[:self.num_references]
        ]

    def _get_references(self, index):
        """
        Get references of a bag

        Parameters
        ----------
        index : int

        Returns
        -------
        list[int]
        """

        refs = []

        for j in range(len(self.bags)):

            if self.ref_matrix[index][j] == 1:
                refs.append(j)

        return refs

    def _get_citers(self, index):
        """
        Get citers of a bag

        Parameters
        ----------
        index : int

        Returns
        -------
        list[int]
        """

        citers = []

        for j in range(len(self.bags)):

            if self.ref_matrix[j][index] == 1:

                citers.append((j, self.D[index][j]))

        citers.sort(key=lambda x: x[1])

        return [
            idx
            for idx, _
            in citers[:self.num_citers]
        ]

    def _get_union_neighbours(self, index):
        """
        Union references + citers

        Parameters
        ----------
        index : int

        Returns
        -------
        list[int]
        """

        refs = self._get_references(index)

        citers = self._get_citers(index)

        return list(set(refs + citers))

    def _calculate_record_label(self, neighbours):
        """
        Count labels in neighbourhood

        Parameters
        ----------
        neighbours : list[int]

        Returns
        -------
        ndarray
        """

        n_labels = self.Y.shape[1]

        counts = np.zeros(n_labels)

        for idx in neighbours:
            counts += self.Y[idx]

        return counts

    def _get_bag_labels(self, index):
        """
        Convert labels to {-1,+1}

        Parameters
        ----------
        index : int

        Returns
        -------
        ndarray
        """

        return np.where(self.Y[index] == 1, 1.0, -1.0)

    def _get_weights_matrix(self):
        """
        Calculate weights matrix

        Returns
        -------
        ndarray
        """

        weights, _, _, _ = np.linalg.lstsq(self.phi_matrix, self.t_matrix, rcond=None)

        return weights

    def _calculate_test_record(self, bag):
        """
        Calculate label-count vector of a test bag

        Parameters
        ----------
        bag : Bag

        Returns
        -------
        ndarray
        """

        n = len(self.bags)

        distances = np.zeros(n)

        for i in range(n):

            distances[i] = self.metric.distance(bag, self.bags[i])

        refs = np.argsort(distances)[:self.num_references]

        citers = []

        for i in range(n):

            train_refs = self._get_references(i)

            if len(train_refs) == 0:
                continue

            worst_reference_distance = max(self.D[i][r]
                for r in train_refs
            )

            if distances[i] < worst_reference_distance:
                citers.append(i)

        if len(citers) > 0:

            citers.sort(key=lambda idx: distances[idx])

            citers = citers[:self.num_citers]

        neighbours = list(set(list(refs) + list(citers)))

        return self._calculate_record_label(neighbours)