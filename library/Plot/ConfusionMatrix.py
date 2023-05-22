import pandas as pd
import seaborn as sn
import plotly.figure_factory as ff
import matplotlib.pyplot as plt


def plot(cm: pd.DataFrame, title):
    ax = plt.axes()
    ax.set_title(title)
    sn.set(font_scale=1.4)
    sn.heatmap(cm, annot=True, annot_kws={"size": 16})
    plt.show()
