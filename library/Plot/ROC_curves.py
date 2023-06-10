import matplotlib.pyplot as plt
from sklearn import metrics





def plot(list_of_label, list_y_predicted, list_y_true):
    """

    :param list_of_label: simple list of label, each one must be a string
    :param list_y_predicted: list of list, each one is a list of the predicted y
    :param list_y_true: ground truth label
    :return:
    """
    for index, label in enumerate(list_of_label):
        fpr, tpr, _ = metrics.roc_curve(list_y_true, list_y_predicted[index], pos_label=1)
        auc = metrics.roc_auc_score(list_y_true, list_y_predicted[index])
        plt.plot(fpr, tpr, label=f"{label}, A.U.C.: {round(auc,2)}")

    plt.legend(loc=4)
    plt.show()
