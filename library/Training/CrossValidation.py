import pandas as pd
import logging, time, os, shutil, warnings, sys
from library.Training.Training import Training
from library.Dataset.DatasetPartition import DatasetPartition
from library.Dataset.Dataset import Dataset
from library.Dataset.EDA import EDA
from library.Plot.Histogram import plot
from library.Plot.EDA_Analysis import plot as EDA_plot


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class CrossValidation:
    """
    This class is used to compare each trained model result in order to find the best combination
    """
    def __init__(self, args):
        """

        :param args: received param. from the script invocation
        """
        self.MODE = args.mode
        self.SOURCE = args.source
        self.VERBOSE = args.verbose
        self.STORE_NORMALIZED = args.store_normalized
        self.MASTER_LOG_PATH = "log/master.log"
        self.LOG_BASE_PATH = "log/testing{}_{}.log"
        self.SUPPORTED_CLASSIFIER = ["LR", "RF", "KNN"]
        self.SUPPORTED_SAMPLING = ["US", "OS_SVM", "OS_K", "OS_ADASYN"]
        self.K_TEST = 1 if self.MODE == 'light' else 5
        self.W_F1 = 5
        self.W_PRECISION = 9
        self.W_RECALL = 3
        self.W_FDR = 7

        self.performances = {}
        self.logger = None
        self.clean_dir()
        self.logger_config()

    def logger_config(self):
        """
        Config. master logger
        :return:
        """
        logging.basicConfig(filename=self.MASTER_LOG_PATH,
                            filemode='w',
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%d-%m-%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger("Master-Logger")
        if self.VERBOSE:
            self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def clean_dir(self):
        """
        Method used to clean all existing log
        :return:
        """
        dir_list = ["log", "classifier"]
        for dir in dir_list:
            for filename in os.listdir(dir):
                file_path = os.path.join(dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)

                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    def execute(self):
        """
        Execute each Model and register each performance
        :return:
        """
        ## Dataset handling
        start = time.time()
        ds = Dataset()
        # OLD -> df = ds.download(save=True)
        df = ds.read_from_file(path=self.SOURCE)
        EDA_plot(df)
        # USED WITH SCALING -> cols = list(df.keys())
        # Dataset normalization & scaling
        eda = EDA(df, verbose=self.VERBOSE)
        eda.replacing(save=self.STORE_NORMALIZED)
        eda.null_val_replace(save=self.STORE_NORMALIZED)
        df = eda.get_df()
        # Dataset partition
        dp = DatasetPartition(df)
        x_training, y_training, x_testing, y_testing = dp.split()
        # NOT SUGGESTED DUE TO LACK OF PERFORMANCE  Scaling training dataset
        # df_training = pd.DataFrame(x_training, columns=cols[:-1])
        # df_training.insert(df_training.shape[1], cols[-1], y_training, True)
        # eda = EDA(df_training, verbose=self.VERBOSE, path_to_save='dataset/data_training_scaled.{}')
        # eda.scaling(save=self.STORE_NORMALIZED)
        # df_training = eda.get_df()
        # dp = DatasetPartition(df_training, split_test=False)
        # x_training, y_training, _, _ = dp.split()
        end = time.time()
        self.logger.info(f"Dataset loaded & analyzed... {round(end - start, 2)}sec")
        # Model training
        start = time.time()
        for classifier in self.SUPPORTED_CLASSIFIER:
            self.performances[classifier] = {}
            for sampler in self.SUPPORTED_SAMPLING:
                self.performances[classifier][sampler] = []
                if self.VERBOSE: self.logger.info(f"Model {classifier} - Sampler {sampler}")
                for index in range(self.K_TEST):
                    start_inner = time.time()
                    t = Training(method=classifier, oversample_tech=sampler,
                                 logging_path=self.LOG_BASE_PATH.format(classifier, sampler))
                    accuracy, f1, fdr, precision, recall = t.run(x_training=x_training, y_training=y_training,
                                                       x_testing=x_testing, y_testing=y_testing,
                                                       mode=self.MODE)
                    # store model performance after internal h_param optimization
                    test_result = {"ACC": accuracy, "F1": f1, "FDR": fdr, "PRECISION": precision, "RECALL": recall}
                    self.performances[classifier][sampler].append(test_result)
                    end_inner = time.time()
                    if self.VERBOSE: self.logger.info(f"{index+1}) Model trained in {round(end_inner - start_inner, 2)}sec...")
        end = time.time()
        self.logger.info(f"All model have been trained in {round(end - start, 2)}sec...")

    def print_result(self):
        """
        Save inside log performances and plot result
        :return:
        """
        ## Model-selection
        self.logger.info("Analyze the best model due to the performances...")
        # creating a dict with all the performances
        result = []
        for classifier in self.SUPPORTED_CLASSIFIER:
            for sampler in self.SUPPORTED_SAMPLING:
                medium_accuracy = round(sum([result["ACC"] for result in self.performances[classifier][sampler]]) / self.K_TEST, 2)
                medium_f1 = round(sum([result["F1"] for result in self.performances[classifier][sampler]]) / self.K_TEST, 2)
                medium_fdr = round(sum([result["FDR"] for result in self.performances[classifier][sampler]]) / self.K_TEST, 2)
                medium_precision = round(
                    sum([result["PRECISION"] for result in self.performances[classifier][sampler]]) / self.K_TEST, 2)
                medium_recall = round(sum([result["RECALL"] for result in self.performances[classifier][sampler]]) / self.K_TEST,
                                      2)
                result.append({"classifier": classifier, "sampler": sampler,
                               "ACC": medium_accuracy,
                               "F1": medium_f1,
                               "FDR": medium_fdr,
                               "PRECISION": medium_precision,
                               "RECALL": medium_recall,
                               "AVG": round(
                                   (self.W_F1 * medium_f1
                                    + self.W_FDR * (100 - medium_fdr)
                                    + self.W_RECALL * medium_recall
                                    + self.W_PRECISION * medium_precision)
                                   / (self.W_F1 + self.W_FDR + self.W_RECALL + self.W_PRECISION),
                                   2)})

        ## Logging all TOP performances / score
        # accuracy
        self.logger.info("ACCURACY..")
        table = sorted(result, key=lambda d: d['ACC'], reverse=True)[:10]
        for row in table:
            self.logger.info('| {:4} | {:10} | {:^4} |'.format(row["classifier"], row["sampler"], row["ACC"]))
        # F1-SCORE
        self.logger.info("F1-SCORE..")
        table = sorted(result, key=lambda d: d['F1'], reverse=True)[:10]
        plot("TOP 5 F1",
             values=[row["F1"] for row in table[:5]],
             labels=[f'{row["classifier"]}_{row["sampler"]}' for row in table[:5]])
        for row in table:
            self.logger.info('| {:4} | {:10} | {:^4} |'.format(row["classifier"], row["sampler"], row["F1"]))
        # FDR
        self.logger.info("FDR...")
        table = sorted(result, key=lambda d: d['FDR'])[:10]
        plot("TOP 5 FDR",
             values=[row["FDR"] for row in table[:5]],
             labels=[f'{row["classifier"]}_{row["sampler"]}' for row in table[:5]])
        for row in table:
            self.logger.info('| {:4} | {:10} | {:^4} |'.format(row["classifier"], row["sampler"], row["FDR"]))
        # RECALL
        self.logger.info("RECALL..")
        table = sorted(result, key=lambda d: d['RECALL'], reverse=True)[:10]
        plot("TOP 5 RECALL",
             values=[row["RECALL"] for row in table[:5]],
             labels=[f'{row["classifier"]}_{row["sampler"]}' for row in table[:5]])
        for row in table:
            self.logger.info('| {:4} | {:10} | {:^4} |'.format(row["classifier"], row["sampler"], row["RECALL"]))
        # PRECISION
        self.logger.info("PRECISION..")
        table = sorted(result, key=lambda d: d['PRECISION'], reverse=True)[:10]
        plot("TOP 5 PRECISION",
             values=[row["PRECISION"] for row in table[:5]],
             labels=[f'{row["classifier"]}_{row["sampler"]}' for row in table[:5]])
        for row in table:
            self.logger.info('| {:4} | {:10} | {:^4} |'.format(row["classifier"], row["sampler"], row["PRECISION"]))

        # GLOBAL SCORE WEIGHTED
        self.logger.info("GLOBAL WEIGHTED SCORE..")
        table = sorted(result, key=lambda d: d['AVG'], reverse=True)[:5]
        plot("TOP 5 GLOBAL",
             values=[row["AVG"] for row in table],
             labels=[f'{row["classifier"]}_{row["sampler"]}' for row in table])
        for row in table:
            self.logger.info('| {:4} | {:10} | {:^4} |'.format(row["classifier"], row["sampler"], row["AVG"]))

