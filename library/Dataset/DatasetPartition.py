import pandas as pd
import numpy as np


class DatasetPartition:
    """This class is used to divide the dataset for training between inference section and training one.
    The Cross-validation method used is 4 Stratification"""

    def __init__(self, df: pd.DataFrame, split_percentage=0.9):
        """
        Constructor is build to split dataset
        :param df: dataset to be split into validation set and training set
        :param split_percentage: declared in case of necessity, not used
        """
        tr_size = int(df.shape[0] * split_percentage)
        self.df_training, self.df_testing = np.split(df, [tr_size], axis=0)
        self.x_training = self.df_training.iloc[:, :len(self.df_training.keys()) - 1].values
        self.y_training = self.df_training.iloc[:, len(self.df_training.keys()) - 1:len(self.df_training.keys())].values
        self.x_testing = self.df_testing.iloc[:, :len(self.df_testing.keys()) - 1].values
        self.y_testing = self.df_testing.iloc[:, len(self.df_testing.keys()) - 1:len(self.df_testing.keys())].values

    def split(self):
        """

        :return: x_training, y_training, x_testing, y_testing
        """
        return self.x_training, self.y_training, self.x_testing, self.y_testing