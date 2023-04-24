import matplotlib.pyplot as plt


def plot(threshold, model):
    """

    :param threshold: data to be plotted
    :param model: label description
    :return: plot
    """
    plt.plot(threshold, linestyle='--', color='red', label=model)
    # title
    plt.title(f'ROC curve {model}')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')