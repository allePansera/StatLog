import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def get_top_abs_correlations(df, n=5):
    df_clone = df.copy()
    df_clone = df_clone.drop(['Target'], axis=1)
    corr_matrix = df_clone.corr().abs()
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
           .stack()
           .sort_values(ascending=False))
    return sol[:n]


def plot(df: pd.DataFrame):
    try:
        fig, ax = plt.subplots(figsize=(15, 5))
        # class distribution bar
        g_b_count = df["Target"].value_counts()[1]
        b_b_count = df["Target"].value_counts()[2]
        ax.title.set_text("Class distribution")
        bars = ax.barh(["Good Borrower", "Bad Borrower"], [g_b_count, b_b_count])
        ax.bar_label(bars)
        plt.show()
        # TOP 5 correlated features
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.title.set_text("Top 5 Correlation")
        top5 = get_top_abs_correlations(df, n=5)
        bars = ax.barh([str(index) for index, val in top5.items()], [round(val,4) for index, val in top5.items()])
        ax.bar_label(bars)
        plt.show()

    except Exception as e:
        raise Exception(f"Error producing EDA plot: {e}")
