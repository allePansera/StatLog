import time, logging, os
from library.Dataset.Dataset import Dataset
from library.Dataset.Normalization import Normalization
from library.Training.RandomForest import RandomForest
from library.Plot.Correlation import plot as correlation_plt
from library.Plot.ConfusionMatrix import plot as conf_matrix_plot
from library.Plot.FeatureImportance import plot as feature_importance


class Training:
    """This class execute the training of the model"""
    def __init__(self, logging_path = 'log/training.log'):
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

            self.logger.info("Classifier production - RandomForest...")
            rf = RandomForest(df)
            start = time.time()
            rf.train()
            cm, f1, good_borrow_precision, bad_borrow_precision = rf.test()
            end = time.time()
            self.logger.info(f"Classifier produced in {round(end - start, 2)}sec")

            conf_matrix_plot(cm)
            feature_importance(rf.classifier.feature_importances_)
            correlation_plt(df)
            rf.save_classifier()

            self.logger.info(f"F1 score: {f1}")
            self.logger.info(f"Good borrower prediction: {round(good_borrow_precision,2)}%")
            self.logger.info(f"Bad borrower prediction: {round(bad_borrow_precision,2)}%")

            self.logger.info(f"Classifier stored...")
            self.logger.info("Training concluded...")
            self.release_logger()
        except Exception as e:
            self.logger.error(f"Error '{e}' while executing training...")
