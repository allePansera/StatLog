import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, recall_score, accuracy_score
from library.Exceptions.CustomExceptions import TrainingException
from sklearn.model_selection import GridSearchCV
from library.Training.Sampler import Sampler
from library.Training.Classifier import Classifier


class RandomForest(Classifier):
    def __init__(self, x_training, y_training, x_testing, y_testing, oversample_tech, mode):
        """
        Constructor split DataSet into training and testing samples
        :param x_training: feature values used for training
        :param y_training: label values used for training
        :param x_testing: feature values used for evaluation
        :param y_testing: label values used for evaluation
        :param oversample_tech: Over-sampling or under-sample technique to use
        :param mode: if hyper param optimization is required
        """
        self.x_training = x_training
        self.y_training = y_training
        self.x_testing = x_testing
        self.y_testing = y_testing
        self.classifier = None
        # UNDERSAMPLE or OVERSAMPLE:
        self.oversample_tech = oversample_tech
        self.mode = mode
        smp = Sampler(oversample_tech=oversample_tech)
        self.x_training, self.y_training = smp.execute(self.x_training, self.y_training)
        # H_PARAM
        self.h_param = {
                    'max_depth': [80, 90, 100, 110],
                    'min_samples_split': [8, 10, 12],
                    'n_estimators': [100, 200, 300, 1000]
                }

    def save_classifier(self, path='classifier/rf.{}'):
        joblib.dump(self.classifier, path.format("joblib"))

    def optimize_hparam(self):
        """
        This method is used to apply h_param optimization
        :return:
        """
        try:
            self.classifier = GridSearchCV(estimator=self.classifier, param_grid=self.h_param, cv=5, n_jobs=-1)
        except Exception as e:
            raise TrainingException(f"Error '{e}' optimizing hpram with LogisticRegression")

    def train(self):
        """
        train() method create the classifier to be used during testing
        :return:
        """
        try:
            self.classifier = RandomForestClassifier()
            if self.mode == 'heavy': self.optimize_hparam()
            self.classifier.fit(self.x_training, self.y_training.ravel())
        except Exception as e:
            raise TrainingException(f"Error '{e}' training dataset with RandomForest classifier")

    def test(self):
        """
        test() check the result for the Random Forest classifier produced
        :return: confusion matrix error, accuracy, f1 score, good borrower precision, bad borrower precision, fpr, precision, roc threshold, model, param scelti
        """
        try:
            if self.classifier is None:
                raise Exception('classifier still not produced')

            model = f"{self.SUPPORTED_METHOD['RF']} - {self.SUPPORTED_SAMPLES[self.oversample_tech]}"
            y_predicted = self.classifier.predict(self.x_testing)
            cm = confusion_matrix(self.y_testing, y_predicted)
            accuracy = accuracy_score(self.y_testing, y_predicted)
            f1 = f1_score(self.y_testing, y_predicted)
            TP = cm[0][0]
            FP = cm[1][0]
            TN = cm[1][1]
            FN = cm[0][1]
            fdr = round(FP/(TP+FP), 2)
            tpr, fpr, threshold = roc_curve(self.y_testing, y_predicted, pos_label=1)
            precision = precision_score(self.y_testing, y_predicted)
            recall = recall_score(self.y_testing, y_predicted)
            return cm, accuracy, f1, fdr, precision, recall, threshold, model, self.classifier.best_params_ if self.mode == 'heavy' else None
        except Exception as e:
            raise TrainingException(f"Error '{e}' testing RandomForest classifier produced")
