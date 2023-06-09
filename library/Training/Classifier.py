import numpy as np
import pandas as pd
from library.Exceptions.CustomExceptions import TrainingException


class Classifier:
    SUPPORTED_METHOD = {"RF": "Random Forest", "LR": "Logical Regression", "KNN": "K-Neighbour"}
    SUPPORTED_SAMPLES = {"US": "Undersample - Near Miss",
                         "OS_K": "Oversample - K SMOTE",
                         "OS_SVM": "Oversample - SVM SMOTE",
                         "OS_ADASYN": "Oversample - ADASYN"}
    """
    Factory method implementation in order to change from code the TRAINING CLASSIFIER
    """

    def __init__(self, x_training, y_training, x_testing, y_testing, method, oversample_tech, mode):
        """
        Constructor split DataSet into training and testing samples
        :param x_training: feature values used for training
        :param y_training: label values used for training
        :param x_testing: feature values used for evaluation
        :param y_testing: label values used for evaluation
        :param method: it's used to decide which classifier train
        :param oversample_tech: Over-sampling or under-sample technique to use
        :param mode: used to decide where hyper param optimization is required or not
        """
        self.method = method
        self.oversample_tech = oversample_tech

        from library.Training.RandomForest import RandomForest
        from library.Training.LogisticRegression import LogisticRegression
        from library.Training.KNeighbour import KNeighbour
        if self.method == "RF":
            self.cl = RandomForest(x_training, y_training, x_testing, y_testing, oversample_tech, mode)

        elif self.method == "LR":
            self.cl = LogisticRegression(x_training, y_training, x_testing, y_testing, oversample_tech, mode)

        elif self.method == "KNN":
            self.cl = KNeighbour(x_training, y_training, x_testing, y_testing, oversample_tech, mode)

        else:
            raise TrainingException(f"Classifier '{self.method}' not supported")

    def train(self):
        """

        :param df: DataFrame used by supported classifiers
        :return: the trained classifier
        """
        return self.cl.train()

    def test(self):
        return self.cl.test()

    def save_classifier(self, path='classifier/rf_{}_{}.{}'):
        return self.cl.save_classifier(path=path)

    def get_classifier(self):
        return self.cl.classifier

