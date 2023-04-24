import numpy as np
import pandas as pd
from library.Exceptions.CustomExceptions import TrainingException


class Classifier:
    SUPPORTED_METHOD = {"RF": "Random Forest", "LR": "Logical Regression"}
    SUPPORTED_SAMPLES = {"US": "Undersample - Near Miss",
                         "OS_K": "Oversample - K SMOTE",
                         "OS_SVM": "Oversample - SVM SMOTE",
                         "OS_ADASYN": "Oversample - ADASYN"}
    """
    Factory method implementation in order to change from code the TRAINING CLASSIFIER
    """
    def __init__(self, df: pd.DataFrame, method, oversample_tech):
        """

        :param df: DataFrame used by the classifier method
        :param method: classifier family
        :param oversample_tech: over-sampling or under-sampling technique
        """
        self.method = method
        from library.Training.RandomForest import RandomForest
        from library.Training.LogicalRegression import LogicalRegression
        if self.method == "RF":
            self.cl = RandomForest(df, oversample_tech)

        elif self.method == "LR":
            self.cl = LogicalRegression(df, oversample_tech)

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

    def save_classifier(self, path='classifier/rf.{}'):
        return self.cl.save_classifier(path=path)

    def get_classifier(self):
        return self.cl.classifier
