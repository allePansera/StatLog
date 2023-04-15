import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from library.Exceptions.CustomExceptions import TrainingException


class RandomForest:
    def __init__(self, df: pd.DataFrame, split_percentage=0.9):
        """
        Constructor split DataSet into training and testing samples
        :param df: DataFrame to use
        :param split_percentage: percentage used to split Dataset between training and testing
        """
        tr_size = int(df.shape[0] * split_percentage)
        self.df_training, self.df_testing = np.split(df, [tr_size], axis=0)
        self.training_undersample()
        self.x_training = self.df_training.iloc[:, :len(self.df_training.keys())-1].values
        self.y_training = self.df_training.iloc[:, len(self.df_training.keys())-1:len(self.df_training.keys())].values
        self.x_testing = self.df_testing.iloc[:, :len(self.df_testing.keys()) - 1].values
        self.y_testing = self.df_testing.iloc[:, len(self.df_testing.keys())-1:len(self.df_testing.keys())].values
        self.classifier = None
        self.max_depth = 50
        self.n_estimators = 100

    def save_classifier(self, path='classifier/rf.{}'):
        joblib.dump(self.classifier, path.format("joblib"))

    def training_undersample(self):
        """
        This method is used to load the n. of samples for each class.

        :return:
        """
        n_max = min(self.df_training['Target'].value_counts()[1], self.df_training['Target'].value_counts()[2])
        self.df_training = self.df_training.groupby('Target').apply(lambda x: x.sample(n=min(n_max, len(x))))

    def train(self):
        """
        train() method create the classifier to be used during testing
        :return:
        """
        try:
            self.classifier = RandomForestClassifier(n_estimators=self.n_estimators,
                                                     max_depth=self.max_depth,
                                                     n_jobs=10,
                                                     criterion="gini",
                                                     class_weight={1: 1, 2: 5})
            self.classifier.fit(self.x_training, self.y_training.ravel())
        except Exception as e:
            raise TrainingException(f"Error '{e}' training dataset with RandomForest classifier")

    def test(self):
        """
        test() check the result for the Random Forest classifier produced
        :return: confusion matrix error, f1 score, good borrower precision, bad borrower precision
        """
        try:
            if self.classifier is None:
                raise Exception('classifier still not produced')

            y_predicted = self.classifier.predict(self.x_testing)
            cm = confusion_matrix(self.y_testing, y_predicted)
            f1 = f1_score(self.y_testing, y_predicted)
            precision_good_credit = (cm[0][0] / (cm[0][0] + cm[0][1])) * 100
            precision_bad_credit = (cm[1][1] / (cm[1][0] + cm[1][1])) * 100
            return cm, f1, precision_good_credit, precision_bad_credit
        except Exception as e:
            raise TrainingException(f"Error '{e}' testing RandomForest classifier produced")
