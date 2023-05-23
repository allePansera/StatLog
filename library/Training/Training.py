import time, logging, os
import traceback

import pandas as pd

from library.Dataset.Dataset import Dataset
from library.Dataset.Normalization import Normalization
from library.Training.Classifier import Classifier
from library.Plot.Correlation import plot as correlation_plt
from library.Plot.ConfusionMatrix import plot as conf_matrix_plot
from library.Plot.FeatureImportance import plot as feature_importance
from library.Plot.RocCurve import plot as roc_curve
from library.Exceptions.CustomExceptions import TrainingException


class Training:
    """This class execute the training of the model"""
    def __init__(self, method="RF", oversample_tech='OS_SVM', logging_path='log/training.log'):
        """
        Initialize the logger that print training status
        :param logging_path: path to write log messages
        """
        # logging.basicConfig(filename=logging_path,
        # filemode='w',
        # format='%(asctime)s %(levelname)-8s %(message)s',
        # datefmt='%d-%m-%Y %H:%M:%S',
        # level=logging.INFO)
        fh = logging.FileHandler(logging_path, 'a', 'utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s %(levelname)-8s %(message)s')
        fh.setFormatter(formatter)
        self.logger = logging.getLogger("TrainingLogger{}{}".format(method, oversample_tech))
        self.release_logger()
        self.logger.addHandler(fh)
        self.logger.propagate = False
        self.method = method
        self.oversample_tech = oversample_tech


    def release_logger(self):
        """
        This method clean logger
        :return: nothing
        """
        if self.logger is not None:
            handlers = self.logger.handlers[:]
            for handler in handlers:
                self.logger.removeHandler(handler)
                handler.close()

    def run(self, x_training, y_training, x_testing, y_testing):
        """
        run() method start training writing output to log file defined as a library attribute
        :param x_training: passed to the classifier
        :param y_training: passed to the classifier
        :param x_testing: passed to the classifier to evaluate performance
        :param y_testing: passed to the classifier to evaluate performance
        :return: F1, FDR, PRECISION, RECALL
        """
        try:
            self.logger.info("Starting...")
            self.logger.info(f"Classifier production - {self.method}...")
            cl = Classifier(x_training=x_training, y_training=y_training,
                            x_testing=x_testing, y_testing=y_testing,
                            method=self.method,
                            oversample_tech=self.oversample_tech)
            start = time.time()
            cl.train()
            cm, f1, good_borrow_precision, bad_borrow_precision, fdr, precision, recall, threshold, model = cl.test()
            end = time.time()
            self.logger.info(f"Classifier produced in {round(end - start, 2)}sec")

            conf_matrix_plot(cm, f'Model: {self.method} - {self.oversample_tech}')
            # roc_curve(threshold, model)
            # if hasattr(cl.get_classifier(), "feature_importances_"):
            #    feature_importance(cl.get_classifier().feature_importances_)

            # USELESS: correlation_plt(df)
            cl.save_classifier(path=f'classifier/rf_{self.method}_{self.oversample_tech}.joblib')

            self.logger.info(f"F1 score: {f1*100}")
            self.logger.info(f"FDR: {round(fdr, 2)*100}%")
            self.logger.info(f"Precision: {round(precision, 2)*100}%")
            self.logger.info(f"Recall: {round(recall, 2) * 100}%")

            self.logger.info("Training concluded...")
            self.release_logger()
            return f1*100, round(fdr, 2)*100, round(precision, 2)*100, round(recall, 2)*100
        except Exception as e:
            self.logger.error(f"Error '{e}' while executing training...")
