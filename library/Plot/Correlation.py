import plotly.express as px
import pandas as pd
from library.Exceptions.CustomExceptions import CorrelationException


def plot(df: pd.DataFrame):
    try:
        fig = px.imshow(df.corr())
        fig.show()
    except Exception as e:
        raise CorrelationException(f"Error '{e}' realizing Correlation Matrix plot")
