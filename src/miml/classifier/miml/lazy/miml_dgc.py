import numpy as np

from .multi_instance_multi_label_knn import MultiInstanceMultiLabelKNN
from ....core.average_hausdorff import AverageHausdorff

class MIMLDGC(MultiInstanceMultiLabelKNN):
    """
    Class to represent Multi instance Multi Label Data Gravitation Classification K Nearest Neighbors
    """
    def __init__(self, num_of_neighbours=10, metric=None, extension=False):
        """
        Constructor of the class MIMLDGCkNN

        Parameters
        ----------
        metric: the distance metric between bags
        num_of_neighbours: number of neighbours
        extension: type of extension (NONE, EXTA, EXTB)
        """

        super().__init__(metric, num_of_neighbours)

        if metric == None:
            metric = AverageHausdorff()
        self.k = num_of_neighbours
        self.metric = metric 
        self.extension = extension

        self.bags = None
        self.Y = None

        self.NGC = None
        self.weights = None
        self.densities = None

        self.weight_max = -np.inf
        self.weight_min = np.inf


    def fit(self, bags):
        """
        Training the classifier

        Parameters
        ----------

        bags: list of Bag
        """

        n = len(bags)

        self.D = self.get_distances(bags)

        self.NGC = np.zeros(n)
        self.weights = np.zeros(n)
        self.densities = np.zeros(n)

        for i in range(n):
            neighbors = self._get_knn(i)
            self._compute_weight_density(i, neighbors)

        for i in range(n):
            if self.weight_max != self.weight_min:
                self.weights[i] = (self.weights[i] - self.weight_min) / (self.weight_max - self.weight_min)
            else:
                self.weights[i] = 0

            self.NGC[i] = self.densities[i] ** self.weights[i]


    def build_internal(self, training_set):
        """
        Method to train dataset

        Parameters
        ----------
        training_set: dataset to train
        """
        self.bags = list(training_set.data.values())

        self.Y = np.array([
            bag.get_labels()[0] for bag in self.bags
        ])

        self.fit(self.bags)


    def make_prediction_internal(self, bag):
        """
        Predict the bag distance with training bag 

        Parameters
        ----------
        bag
        """
        distances = []

        for i in range(len(self.bags)):
            d = self.metric.distance(bag, self.bags[i])
            distances.append((i, d))

        distances.sort(key=lambda x: x[1])

        if not self.extension:
            neighbors = distances[:self.k]
        else:
            kth_dist = distances[self.k - 1][1]
            neighbors = [pair for pair in distances if pair[1] <= kth_dist]

        k = len(neighbors)

        gforce = np.zeros(k)

        for idx, (i, d) in enumerate(neighbors):
            if d == 0:
                d = 1e-10
            gforce[idx] = self.NGC[i] / (d ** 2)

        n_labels = self.Y.shape[1]
        bipartition = np.zeros(n_labels, dtype=int)
        confidence = np.zeros(n_labels)

        for l in range(n_labels):
            pos = 0
            neg = 0

            for idx, (i, _) in enumerate(neighbors):
                if self.Y[i][l] == 1:
                    pos += gforce[idx]
                else:
                    neg += gforce[idx]

            if pos > neg:
                bipartition[l] = 1

            if (pos + neg) > 0:
                confidence[l] = pos / (pos + neg)

        return bipartition, confidence

    def _get_knn(self, i):
        """
        Get k neareast neighbours between bags

        ParametersParameters:
        ----------
        i: number of instance
        """
        distances = []

        for j in range(len(self.bags)):
            if i == j:
                continue

            d = self.D[i][j]
            #d = self.metric.distance(self.bags[i], self.bags[j])
            distances.append((j, d))

        distances.sort(key=lambda x: x[1])

        if not self.extension:
            return [idx for idx, _ in distances[:self.k]]
        else:
            kth = distances[self.k - 1][1]
            return [idx for idx, d in distances if d <= kth]
        
        
    def _label_distance(self, i, j):
        """
        Get distance between labels

        Parameters
        ----------
        i, j: number of instances
        """
        return np.mean(self.Y[i] != self.Y[j])
    

    def _compute_weight_density(self, i, neighbors):
        """
        Set compute neighbours density and weight of an instance
        
        Parameters
        ----------
        i: number of instance
        neighbors: list of neighbours
        """
        weight = 1
        density = 0

        PdisY = 0
        PdisF = 0
        PdisY_disF = 0

        k = len(neighbors)

        for j in neighbors:
            dl = self._label_distance(i, j)
            df = self.metric.distance(self.bags[i], self.bags[j])

            if df == 0:
                continue 

            density += (1 - dl) / df

            PdisY += dl
            PdisF += df
            PdisY_disF += dl * df

        density = 1 + density

        PdisY /= k
        PdisF /= k
        PdisY_disF /= k

        if PdisY == 0 or PdisY == 1:
            weight = 0
        else:
            weight = ((PdisY_disF * PdisF) / PdisY) - (
                ((1 - PdisY_disF) * PdisF) / (1 - PdisY)
            )

        self.weight_max = max(self.weight_max, weight)
        self.weight_min = min(self.weight_min, weight)

        self.weights[i] = weight
        self.densities[i] = density

    def get_bag_labels(bag):
        return np.max(bag.get_labels(), axis=0)