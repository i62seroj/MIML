import numpy as np
from collections import Counter

from .multi_instance_multi_label_knn import MultiInstanceMultiLabelKNN
from ....core.average_hausdorff import AverageHausdorff


class MIMLMAPkNN(MultiInstanceMultiLabelKNN):

    def __init__(self, num_of_neighbours=10, metric=None, smooth=1.0):

        super().__init__(num_of_neighbours, metric)

        if metric == None:
            metric = AverageHausdorff()

        self.k = num_of_neighbours
        self.metric = metric
        self.smooth = smooth

        self.bags = None
        self.Y = None

        self.D = None
        self.prior = None
        self.prior_n = None

        self.cond = None
        self.cond_n = None

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
        self.prior_n = np.zeros(n_labels)

        self.cond = np.zeros((n_labels, self.k + 1))
        self.cond_n = np.zeros((n_labels, self.k + 1))

        self._compute_prior(n, n_labels)
        self._compute_conditional(n, n_labels)
        
        self.trained = True

    def _compute_prior(self, n, n_labels):
        """
        Compute prior probabilities.
        """

        for l in range(n_labels):

            positives = np.sum(self.Y[:, l])

            self.prior[l] = (self.smooth + positives) / (self.smooth * 2 + n)

            self.prior_n[l] = 1.0 - self.prior[l]

    def _compute_conditional(self, n, n_labels):
        """
        Compute conditional probabilities.
        """

        temp_c = np.zeros((n_labels, self.k + 1), dtype=int)

        temp_nc = np.zeros((n_labels, self.k + 1), dtype=int)

        for i in range(n):

            neighbors = self._get_neighbors(i)

            for l in range(n_labels):

                aces = 0

                for j in neighbors:

                    if self.Y[j][l] == 1:
                        aces += 1

                if self.Y[i][l] == 1:
                    temp_c[l][aces] += 1
                else:
                    temp_nc[l][aces] += 1

        for l in range(n_labels):

            total_c = np.sum(temp_c[l])
            total_nc = np.sum(temp_nc[l])

            for j in range(self.k + 1):

                self.cond[l][j] = (self.smooth + temp_c[l][j]) / (self.smooth * (self.k + 1) + total_c)

                self.cond_n[l][j] = (self.smooth + temp_nc[l][j]) / (self.smooth * (self.k + 1) + total_nc)



    def _get_neighbors(self, i):
        """
        Get k nearest neighbours of training bag i.
        """

        distances = []

        for j in range(len(self.bags)):

            if i == j:
                continue

            distances.append((j, self.D[i][j]))

        distances.sort(
            key=lambda x: x[1]
        )

        return [
            idx
            for idx, _ in distances[:self.k]
        ]

    def predict(self, bag):
        
        distances = []

        for i in range(len(self.bags)):

            d = self.metric.distance(bag,self.bags[i])

            distances.append((i, d))

        distances.sort(
            key=lambda x: x[1]
        )

        neighbors = distances[:self.k]

        n_labels = self.Y.shape[1]

        prediction = np.zeros(n_labels, dtype=int)

        confidence = np.zeros(n_labels, dtype=float)

        for l in range(n_labels):

            aces = 0

            for idx, _ in neighbors:

                if self.Y[idx][l] == 1:
                    aces += 1

            p1 = (self.prior[l] * self.cond[l][aces])
            p0 = (self.prior_n[l] * self.cond_n[l][aces])

            if p1 > p0:
                prediction[l] = 1

            if (p1 + p0) > 0:
                confidence[l] = (p1 / (p1 + p0))

        return prediction, confidence
    
    def evaluate(self, dataset_test):

        if not self.trained:
            raise Exception(
                "The classifier is not trained. You need to call fit before predict anything"
            )

        test_bags = list(dataset_test.data.values())

        predictions = []

        for bag in test_bags:

            prediction, _ = self.predict(bag)

            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, dataset_test):

        if not self.trained:
            raise Exception(
                "The classifier is not trained. You need to call fit before predict anything"
            )

        test_bags = list(dataset_test.data.values())

        probabilities = []

        for bag in test_bags:

            _, confidence = self.predict(bag)

            probabilities.append(confidence)

        return np.array(probabilities)