import matplotlib.pyplot as plt
import pandas as pd


def plot(title, values, labels):
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.title.set_text(title)
        bars = ax.barh(labels, values)
        ax.bar_label(bars)
        # ax.set_xticklabels(labels, rotation=90)
        plt.show()
    except Exception as e:
        raise Exception(f"Error '{e}' realizing Histogram plot")
