import numpy as np
from collections import defaultdict

from ....core.average_hausdorff import AverageHausdorff
from .multi_instance_multi_label_knn import MultiInstanceMultiLabelKNN


class MLkNN(MultiInstanceMultiLabelKNN):

    def __init__(self, k=10, metric=None, smooth=1.0):

        if metric is None:
            metric = AverageHausdorff()

        super().__init__(metric, k)

        self.k = k
        self.metric = metric
        self.smooth = smooth

        self.bags = None
        self.Y = None

        self.D = None
        self.prior = None
        self.cond = None


    def build_internal(self, training_set):

        self.bags = list(training_set.data.values())

        self.Y = np.array([
            np.max(bag.get_labels(), axis=0)
            for bag in self.bags
        ])

        self.fit(self.bags)


    def fit(self, bags):

        n = len(bags)

        self.D = self.get_distances(bags)

        n_labels = self.Y.shape[1]

        self.prior = np.zeros(n_labels)

        for l in range(n_labels):

            count = np.sum(self.Y[:, l])

            self.prior[l] = (
                self.smooth + count
            ) / (
                self.smooth * 2 + n
            )

        self.cond = np.zeros((n_labels, self.k + 1))

        temp = np.zeros((n_labels, self.k + 1))

        for i in range(n):

            neighbors = self._get_neighbors(i)

            for l in range(n_labels):

                aces = 0

                for j in neighbors:

                    if self.Y[j][l] == 1:
                        aces += 1

                if self.Y[i][l] == 1:
                    temp[l][aces] += 1

        for l in range(n_labels):

            total = np.sum(temp[l])

            for j in range(self.k + 1):

                self.cond[l][j] = (
                    self.smooth + temp[l][j]
                ) / (
                    self.smooth * (self.k + 1) + total
                )


    def _get_neighbors(self, i):

        distances = [
            (j, self.D[i][j])
            for j in range(len(self.bags))
            if i != j
        ]

        distances.sort(key=lambda x: x[1])

        return [idx for idx, _ in distances[:self.k]]

    def make_prediction_internal(self, bag):

        distances = []

        for i in range(len(self.bags)):

            d = self.metric.distance(bag, self.bags[i])

            distances.append((i, d))

        distances.sort(key=lambda x: x[1])

        neighbors = distances[:self.k]

        n_labels = self.Y.shape[1]

        prediction = np.zeros(n_labels)

        for l in range(n_labels):

            aces = 0

            for idx, _ in neighbors:

                if self.Y[idx][l] == 1:
                    aces += 1

            p1 = self.prior[l] * self.cond[l][aces]
            p0 = (1 - self.prior[l]) * self.cond[l][aces]

            prediction[l] = 1 if p1 > p0 else 0

        return prediction