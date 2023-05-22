import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, recall_score
from library.Exceptions.CustomExceptions import TrainingException
from library.Training.Sampler import Sampler
from library.Training.Classifier import Classifier


class LogicalRegression(Classifier):
    def __init__(self, x_training, y_training, x_testing, y_testing, oversample_tech):
        """
        Constructor split DataSet into training and testing samples
        :param x_training: feature values used for training
        :param y_training: label values used for training
        :param x_testing: feature values used for evaluation
        :param y_testing: label values used for evaluation
        :param oversample_tech: Over-sampling or under-sample technique to use
        """
        self.x_training = x_training
        self.y_training = y_training
        self.x_testing = x_testing
        self.y_testing = y_testing
        self.classifier = None
        self.max_depth = 50
        self.n_estimators = 100
        # UNDERSAMPLE or OVERSAMPLE:
        self.oversample_tech = oversample_tech
        smp = Sampler(oversample_tech=oversample_tech)
        self.x_training, self.y_training = smp.execute(self.x_training, self.y_training)

    def save_classifier(self, path='classifier/rf.{}'):
        joblib.dump(self.classifier, path.format("joblib"))

    def train(self):
        """
        train() method create the classifier to be used during testing
        :return:
        """
        try:
            self.classifier = LogisticRegression(random_state=0, solver='newton-cholesky')
            self.classifier.fit(self.x_training, self.y_training.ravel())
        except Exception as e:
            raise TrainingException(f"Error '{e}' training dataset with RandomForest classifier")

    def test(self):
        """
        test() check the result for the Random Forest classifier produced
        :return: confusion matrix error, f1 score, good borrower precision, bad borrower precision, fpr, precision, threshold, model
        """
        try:
            if self.classifier is None:
                raise Exception('classifier still not produced')

            model = f"{self.SUPPORTED_METHOD['LR']} - {self.SUPPORTED_SAMPLES[self.oversample_tech]}"
            y_predicted = self.classifier.predict(self.x_testing)
            cm = confusion_matrix(self.y_testing, y_predicted)
            f1 = f1_score(self.y_testing, y_predicted)
            precision_good_credit = (cm[0][0] / (cm[0][0] + cm[0][1])) * 100
            precision_bad_credit = (cm[1][1] / (cm[1][0] + cm[1][1])) * 100
            tpr, fpr, threshold = roc_curve(self.y_testing, y_predicted, pos_label=1)
            precision = precision_score(self.y_testing, y_predicted)
            recall = recall_score(self.y_testing, y_predicted)
            return cm, f1, precision_good_credit, precision_bad_credit, fpr[1], precision, recall, threshold, model
        except Exception as e:
            raise TrainingException(f"Error '{e}' testing RandomForest classifier produced")
