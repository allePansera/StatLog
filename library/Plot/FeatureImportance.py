import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot(feature_importance):
    indices = np.argsort(feature_importance)[::-1]
    indices = indices[:10]
    plt.figure()
    plt.title("Top 10 Feature importance")
    plt.bar(range(10), feature_importance[indices], color="r", align="center")
    plt.xticks(range(10), indices)
    plt.xlim([-1, 10])
    plt.show()