import pandas as pd


class Dataset:
    """This class is used to import a specific dataset from URL.
    The default URL is Statlog (German Credit Data)"""

    def __init__(self, url='https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'):
        """
        Constructor set the url as class attribute.
        :param url: url to use in order to download the dataset
        """

        self.url = url

    def download(self):
        """
        Method download dataset from class attribute self.url.
        read_csv() parameters are defined for UCI specific dataset.
        :return: DataFrame object built after read_csv method call
        """
        return pd.read_csv(self.url)

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
        df.to_csv(path.format('csv'), index=False)
        df.to_excel(path.format('xlsx'), index=True)


d = Dataset()
df = d.download()
d.store_dataframe(df)
