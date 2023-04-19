import numpy as np
from imblearn.over_sampling import SVMSMOTE, KMeansSMOTE, ADASYN
from imblearn.under_sampling import NearMiss


class Sampler:

    def __init__(self, oversample_tech):
        self.oversample_tech = oversample_tech

    def execute(self, x_training: np.ndarray, y_training: np.ndarray):
        """
        :return: x_training, y_training updated
        """
        if self.oversample_tech == "US":
            return self.training_undersample(x_training, y_training)
        elif self.oversample_tech == "OS_K":
            return self.training_oversample_k_smote(x_training, y_training)
        elif self.oversample_tech == "OS_SVM":
            return self.training_oversample_svm_smote(x_training, y_training)
        elif self.oversample_tech == "OS_ADASYN":
            return self.training_oversample_adasyn(x_training, y_training)

    def training_undersample(self, x_training: np.ndarray, y_training: np.ndarray):
        """
        This method is used to load the n. of samples for each class. NearMiss technique is used.
        :return: x_training, y_training updated
        """
        nm = NearMiss()
        x_training, y_training = nm.fit_resample(x_training, y_training)
        return x_training, y_training

    def training_oversample_k_smote(self, x_training: np.ndarray, y_training: np.ndarray):
        """
        This method perform K_SMOTE Technique un order to generate more samples to balance class.
        :return: Nothing
        """
        oversample = KMeansSMOTE()
        x_training, y_training = oversample.fit_resample(x_training, y_training)
        return x_training, y_training

    def training_oversample_svm_smote(self, x_training: np.ndarray, y_training: np.ndarray):
        """
        This method perform SVM_SMOTE Technique un order to generate more samples to balance class.
        :return: Nothing
        """
        oversample = SVMSMOTE()
        x_training, y_training = oversample.fit_resample(x_training, y_training)
        return x_training, y_training

    def training_oversample_adasyn(self, x_training: np.ndarray, y_training: np.ndarray):
        """
        This method perform ADASYN Technique un order to generate more samples to balance class.
        :return: Nothing
        """
        oversample = ADASYN()
        x_training, y_training = oversample.fit_resample(x_training, y_training)
        return x_training, y_training