import pandas as pd
import requests


class Dataset:
    """This class is used to import a specific dataset from URL.
    The default URL is Statlog (German Credit Data)"""

    def __init__(self, url='https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'):
        """
        Constructor set the url as library attribute.
        :param url: url to use in order to download the dataset
        """

        self.url = url

    def download(self, save=False):
        """
        Method download dataset from library attribute self.url.
        read_csv() parameters are defined for UCI specific dataset.
        :param save: is True DataFrame downloaded is stored as a file.
        :return: DataFrame object built after read_csv method call
        """
        names = ["Checking account",
                 "Duration",
                 "Credit history",
                 "Purpose",
                 "Credit amount",
                 "Savings account",
                 "Employment since",
                 "Installment rate",
                 "Status & sex",
                 "Debtors & guarantors",
                 "Residence since",
                 "Property",
                 "Age",
                 "Other installments",
                 "Housing",
                 "Existing credits",
                 "Job",
                 "Kept people",
                 "Phone",
                 "Foreign worker", "Target"]
        res = requests.get(self.url)
        data = [row.split(" ") for row in res.content.decode().split('\n')]
        df = pd.DataFrame(data=data[:-1], index=range(len(data)-1), columns=names)
        df = df.astype(str)
        if save:
            self.store_dataframe(df)
        return df

    def store_dataframe(self, df, path='./dataset/data.{}'):
        """
        Method store a dataframe inside a Path. DataFrame object is stored as CSV and XLSX.
        Path format is checked as df type.
        :param df: Dataframe instance to save as file.
        :param path: path where DataFrame is saved. Format must be omitted.
        :return: Nothing
        """
        if '.{}' not in path:
            raise Exception("Wrong 'path' parameter. Do not specify file type, use {} instead.")
        if not isinstance(df, pd.DataFrame):
            raise Exception("'df' parameter must be a Dataframe instance.")
        df.to_csv(path.format('csv'), index=True)
        df.to_excel(path.format('xlsx'), merge_cells=False, index=True)


