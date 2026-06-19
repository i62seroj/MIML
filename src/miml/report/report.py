import warnings

import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, fbeta_score, jaccard_score, log_loss, \
    roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from ..data import MIMLDataset


class Report:
    """
    Class to generate a report
    """

    def __init__(self, y_pred: np.ndarray, label_probs: np.ndarray, dataset_test: MIMLDataset,
                 metrics: list[str] = None, header: bool = True, per_label: bool = False):
        """
        Constructor of the class report
        """
        self.dataset = dataset_test
        self.y_true = dataset_test.get_labels_by_bag()
        self.y_pred = y_pred
        self.probs = label_probs

        all_metrics = ["precision-score-macro", "precision-score-micro", "average-precision-score-macro",
                       "average-precision-score-micro", "recall-score-macro", "recall-score-micro", "f1-score-macro",
                       "f1-score-micro", "fbeta-score-macro", "fbeta-score-micro", "subset-accuracy-score",
                       "accuracy-score", "hamming-loss", "jaccard-score-macro", "jaccard-score-micro", "log-loss"]
        if per_label:
            per_label_metrics = ["precision-score", "recall-score", "f1-score", "fbeta-score", "jaccard-score"]
            for metric in per_label_metrics:
                for label in dataset_test.get_labels_name():
                    all_metrics.append(metric + "-" + label)

        if metrics is None:
            metrics = all_metrics
        else:
            for metric in metrics:
                if metric not in all_metrics:
                    raise Exception("Metric ", metric, "is not valid\n", "Metrics available: ", all_metrics)
        self.header = header
        self.metrics_name = metrics
        self.per_label = per_label
        self.metrics_value = dict()
        self.beta = 0.5
        self.calculate_metrics()

    def calculate_metrics(self, beta: int = 0.5):
        """
        Calculate metrics of the predicted data

        Parameters
        ----------
        beta : int, default = 0.5
            Beta value for the fbeta_score function
        """
        self.beta = beta
        self.metrics_value["precision-score-macro"] = precision_score(self.y_true, self.y_pred, average="macro",
                                                                      zero_division=0)
        self.metrics_value["precision-score-micro"] = precision_score(self.y_true, self.y_pred, average="micro",
                                                                      zero_division=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # When average_precision_score called raise this warming:
            # UserWarning: No positive class found in y_true, recall is set to one for all thresholds.
            self.metrics_value["average-precision-score-macro"] = average_precision_score(self.y_true, self.probs,
                                                                                          average="macro")
            self.metrics_value["average-precision-score-micro"] = average_precision_score(self.y_true, self.probs,
                                                                                          average="micro")
        self.metrics_value["recall-score-macro"] = recall_score(self.y_true, self.y_pred, average="macro",
                                                                zero_division=0)
        self.metrics_value["recall-score-micro"] = recall_score(self.y_true, self.y_pred, average="micro",
                                                                zero_division=0)
        self.metrics_value["f1-score-macro"] = f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        self.metrics_value["f1-score-micro"] = f1_score(self.y_true, self.y_pred, average="micro", zero_division=0)
        self.metrics_value["fbeta-score-macro"] = fbeta_score(self.y_true, self.y_pred, beta=beta, average="macro",
                                                              zero_division=0)
        self.metrics_value["fbeta-score-micro"] = fbeta_score(self.y_true, self.y_pred, beta=beta, average="micro",
                                                              zero_division=0)
        # TODO: ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
        # self.metrics_value["roc-auc-score-macro"] = roc_auc_score(self.y_true, self.probs, average="macro")
        # self.metrics_value["roc-auc-score-micro"] = roc_auc_score(self.y_true, self.probs, average="micro")
        self.metrics_value["subset-accuracy-score"] = accuracy_score(self.y_true, self.y_pred)
        self.metrics_value["accuracy-score"] = hamming_accuracy_score(self.y_true, self.y_pred)
        self.metrics_value["hamming-loss"] = hamming_loss(self.y_true, self.y_pred)
        self.metrics_value["jaccard-score-macro"] = jaccard_score(self.y_true, self.y_pred, average="macro",
                                                                  zero_division=0)
        self.metrics_value["jaccard-score-micro"] = jaccard_score(self.y_true, self.y_pred, average="micro",
                                                                  zero_division=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # When log_loss called raise this warning, it is not true for multilabel classification
            # UserWarning: The y_pred values do not sum to one. Starting from 1.5 this will result in an error.
            self.metrics_value["log-loss"] = log_loss(self.y_true, self.probs)

        if self.per_label:
            precision_score_per_label = list(precision_score(self.y_true, self.y_pred, average=None, zero_division=0))
            recall_score_per_label = list(recall_score(self.y_true, self.y_pred, average=None, zero_division=0))
            f1_score_per_label = list(f1_score(self.y_true, self.y_pred, average=None, zero_division=0))
            fbeta_score_per_label = list(fbeta_score(self.y_true, self.y_pred, beta=beta, average=None, zero_division=0))
            # roc_auc_score_per_label = list(roc_auc_score(self.y_true, self.probs, average=None))
            jaccard_score_per_label = list(jaccard_score(self.y_true, self.y_pred, average=None, zero_division=0))
            for i, label in enumerate(self.dataset.get_labels_name()):
                self.metrics_value["precision-score-" + label] = precision_score_per_label[i]
                self.metrics_value["recall-score-" + label] = recall_score_per_label[i]
                self.metrics_value["f1-score-" + label] = f1_score_per_label[i]
                self.metrics_value["fbeta-score-" + label] = fbeta_score_per_label[i]
                # self.metrics_value["roc-auc-score-"+label] = roc_auc_score_per_label[i]
                self.metrics_value["jaccard-score-" + label] = jaccard_score_per_label[i]

    def to_csv(self, path: str = None, metrics: list[str] = None):
        """
        Print/save data as csv format

        Parameters
        ----------
        path : str, default=None
            Path to csv where the data would be stored

        metrics : list[str], default=None
            List of metrics to show. If empty, show all metrics.
        """
        # If not metrics passed, show all metrics
        if metrics is None:
            metrics = self.metrics_name

        header = ""
        if self.header:
            for metric in range(len(metrics)):
                header += metrics[metric]
                if metrics[metric][0:5] == "fbeta":
                    header += " beta value = " + str(self.beta)
                header += ","
            header = header[0:-1]

        values = ",".join(str(self.metrics_value[metric]) for metric in metrics)

        # If not path passed, print on screen
        if path is None:
            print(header)
            print(values)
        # Else, save it in a file
        else:
            with open(path, mode="a") as f:
                f.write(header)
                f.write(values)

    def to_string(self, metrics: list[str] = None):
        """
        Print data as string format

        Parameters
        ----------
        metrics : list[str], default=None
            List of metrics to show. If empty, show all metrics.
        """
        # If not metrics passed, show all metrics
        if metrics is None:
            metrics = self.metrics_name

        for metric in metrics:
            if metric[0:5] == "fbeta":
                print(metric, "beta value =", self.beta, ": ", self.metrics_value[metric])
            print(metric, ": ", self.metrics_value[metric])


def hamming_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate hamming accuracy score of given data

    Parameters
    ----------
    y_true : np.ndarray of shape (n_bags, n_labels)
       Labels from the test dataset

    y_pred : np.ndarray of shape (n_bags, n_labels)
       Predicted labels from the test dataset
    """
    numerator = (y_true.astype(int) & y_pred.astype(int)).sum(axis=1)
    denominator = (y_true.astype(int) | y_pred.astype(int)).sum(axis=1)

    return np.divide(numerator, denominator, out=np.ones_like(numerator, dtype=np.float64),
                     where=denominator != 0).mean()
