import matplotlib.pyplot as plt
import pandas as pd
from library.Exceptions.CustomExceptions import CorrelationException


def plot(df: pd.DataFrame):
    try:
        plt.matshow(df.corr())
        plt.show()
    except Exception as e:
        raise CorrelationException(f"Error '{e}' realizing Correlation Matrix plot")
