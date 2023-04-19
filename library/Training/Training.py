import time, logging, os
import traceback

from library.Dataset.Dataset import Dataset
from library.Dataset.Normalization import Normalization
from library.Training.Classifier import Classifier
from library.Plot.Correlation import plot as correlation_plt
from library.Plot.ConfusionMatrix import plot as conf_matrix_plot
from library.Plot.FeatureImportance import plot as feature_importance
from library.Exceptions.CustomExceptions import TrainingException


class Training:
    SUPPORTED_METHOD = ["RF", "LR"]
    SUPPORTED_SAMPLES = ["US", "OS_K", "OS_SVM", "OS_ADASYN"]
    """This class execute the training of the model"""
    def __init__(self, method="RF", oversample_tech='OS_SVM', logging_path='log/training.log'):
        """
        Initialize the logger that print training status
        :param logging_path: path to write log messages
        """

        logging.basicConfig(filename=logging_path,
                            filemode='w',
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%d-%m-%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger("TrainingLogger")
        self.method = method
        self.oversample_tech = oversample_tech
        if self.method not in Training.SUPPORTED_METHOD:
            raise TrainingException(f"Method '{self.method}' not supported. Pick one in {Training.SUPPORTED_METHOD}")
        if self.oversample_tech not in Training.SUPPORTED_SAMPLES:
            raise TrainingException(f"Sample tech. '{self.oversample_tech}' not supported. Pick one in {Training.SUPPORTED_SAMPLES}")

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

    def run(self):
        """
        run() method start training writing output to log file defined as a library attribute
        :return: nothing
        """
        try:
            self.logger.info("Starting...")

            self.logger.info("Downloading dataset...")
            start = time.time()
            ds = Dataset()
            df = ds.download(save=True)
            end = time.time()
            self.logger.info(f"Download concluded in {round(end-start,2)}sec")

            self.logger.info("Normalizing dataset...")
            normalizer = Normalization(df)
            start = time.time()
            df = normalizer.execute(save=True)
            end = time.time()
            self.logger.info(f"Normalization concluded in {round(end - start, 2)}sec")

            self.logger.info(f"Classifier production - {self.method}...")
            cl = Classifier(df, self.method, self.oversample_tech)
            start = time.time()
            cl.train()
            cm, f1, good_borrow_precision, bad_borrow_precision, fpr, precision = cl.test()
            end = time.time()
            self.logger.info(f"Classifier produced in {round(end - start, 2)}sec")

            conf_matrix_plot(cm)
            if hasattr(cl.get_classifier(), "feature_importances_"):
                feature_importance(cl.get_classifier().feature_importances_)
            # USELESS: correlation_plt(df)
            cl.save_classifier()

            self.logger.info(f"F1 score: {f1}")
            self.logger.info(f"Good borrower prediction: {round(good_borrow_precision,2)}%")
            self.logger.info(f"Bad borrower prediction: {round(bad_borrow_precision,2)}%")
            self.logger.info(f"FPR: {round(fpr, 2)*100}%")
            self.logger.info(f"Precision: {round(precision, 2)*100}%")

            self.logger.info(f"Classifier stored...")
            self.logger.info("Training concluded...")
            self.release_logger()
        except Exception as e:
            self.logger.error(f"Error '{e}' while executing training...")
