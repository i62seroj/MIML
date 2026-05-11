import numpy as np
from collections import Counter

from .ml_knn import MLkNN


class MIMLMAPkNN(MLkNN):

    def __init__(self, k=10, metric=None, smooth=1.0):

        super().__init__(k, metric, smooth)


    def make_prediction_internal(self, bag):

        distances = []

        for i in range(len(self.bags)):

            d = self.metric.distance(bag, self.bags[i])

            distances.append((i, d))

        distances.sort(key=lambda x: x[1])

        neighbors = distances[:self.k]


        labelsets = []

        for idx, _ in neighbors:

            labelsets.append(tuple(self.Y[idx].astype(int)))

        counts = Counter(labelsets)

        best_labelset = None
        best_score = -np.inf

        for labelset, count in counts.items():

            prior = self._labelset_prior(labelset)

            likelihood = count / self.k

            score = prior * likelihood

            if score > best_score:

                best_score = score
                best_labelset = labelset

        return np.array(best_labelset)

    def _labelset_prior(self, labelset):

        matches = np.sum([
            np.all(self.Y[i] == labelset)
            for i in range(len(self.Y))
        ])

        return (matches + self.smooth) / (
            len(self.Y) + self.smooth
        )