import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from library.Exceptions.CustomExceptions import TrainingException


class RandomForest:
    def __init__(self, df: pd.DataFrame, split_percentage=0.8):
        """
        Constructor split DataSet into training and testing samples
        :param df: DataFrame to use
        :param split_percentage: percentage used to split Dataset between training and testing
        """
        self.df = df
        self.x = self.df.iloc[:, 1:len(self.df.keys())-1].values
        self.y = self.df.iloc[:, len(self.df.keys())-1:len(self.df.keys())].values
        tr_size = int(self.df.shape[0] * split_percentage)
        self.x_training, self.x_testing = self.x[1:tr_size], self.x[tr_size:]
        self.y_training, self.y_testing = self.y[1:tr_size], self.y[tr_size:]
        self.classifier = None

    def train(self):
        """
        train() method create the classifier to be used during testing
        :return:
        """
        try:
            self.classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")
            self.classifier.fit(self.x_training, self.y_training)
        except Exception as e:
            raise TrainingException(f"Error '{e}' training dataset with RandomForest classifier")

    def test(self):
        """
        test() check the result for the Random Forest classifier produced
        :return: percentage error
        """
        try:
            if self.classifier is None:
                raise Exception('classifier still not produced')

            y_predicted = self.classifier.predict(self.x_testing)
        except Exception as e:
            raise TrainingException(f"Error '{e}' testing RandomForest classifier produced")
