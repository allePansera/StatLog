from library.Training.Training import Training
from library.Dataset.DatasetPartition import DatasetPartition
from library.Dataset.Dataset import Dataset
from library.Dataset.Normalization import Normalization
import logging, time, os, shutil

## Clean all-log
folder = 'log'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

## Logger config
logging.basicConfig(filename="log/master.log",
                            filemode='w',
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%d-%m-%Y %H:%M:%S',
                            level=logging.INFO)
logger = logging.getLogger("Master-Logger")

## Const definition
SUPPORTED_CLASSIFIER = ["RF", "LR", "KNN"]
SUPPORTED_SAMPLING = ["US", "OS_SVM", "OS_K", "OS_ADASYN"]
K_TEST = 10
LOG_BASE_PATH = "log/testing{}_{}.log"
performances = {}




## Dataset division
start = time.time()
# Dataset download
ds = Dataset()
df = ds.download(save=True)
# Dataset normalization
normalizer = Normalization(df)
df = normalizer.execute(save=True)
# Dataset partition
dp = DatasetPartition(df)
x_training, y_training, x_testing, y_testing = dp.split()
end = time.time()
logger.info(f"Dataset downloaded & split between training and validation... {round(end-start,2)}sec")


## Cross-Validation performance
start = time.time()
for classifier in SUPPORTED_CLASSIFIER:
    performances[classifier] = {}
    for sampler in SUPPORTED_SAMPLING:
        performances[classifier][sampler] = []
        for index in range(K_TEST):
            t = Training(method=classifier, oversample_tech=sampler, logging_path=LOG_BASE_PATH.format(classifier, sampler))
            f1, fdr, precision, recall = t.run(x_training=x_training, y_training= y_training,
                                               x_testing=x_testing, y_testing=y_testing)
            test_result = {"F1": f1, "FDR": fdr, "PRECISION": precision, "RECALL": recall}
            performances[classifier][sampler].append(test_result)
end = time.time()
logger.info(f"All model have been trained in {round(end-start, 2)}sec...")

## Model-selection
logger.info("Analyze the best model due to the performances...\n")
# creating a dict with all the performances
result = []
for classifier in SUPPORTED_CLASSIFIER:
    for sampler in SUPPORTED_SAMPLING:
        medium_f1 = round(sum([result["F1"] for result in performances[classifier][sampler]])/K_TEST, 2)
        medium_fdr = round(sum([result["FDR"] for result in performances[classifier][sampler]])/K_TEST, 2)
        medium_precision = round(sum([result["PRECISION"] for result in performances[classifier][sampler]])/K_TEST, 2)
        medium_recall = round(sum([result["RECALL"] for result in performances[classifier][sampler]])/K_TEST, 2)
        result.append({"classifier": classifier, "sampler": sampler,
                       "F1": medium_f1,
                       "FDR": medium_fdr,
                       "PRECISION": medium_precision,
                       "RECALL": medium_recall})

## Logging all TOP 3 performances per each evaluation variable
# F1-SCORE
logger.info("F1-SCORE..")
table = sorted(result, key=lambda d: d['F1'], reverse=True)[:10]
for row in table:
    logger.info('| {:4} | {:10} | {:^4} |'.format(row["classifier"], row["sampler"], row["F1"]))
# FDR
logger.info("FDR...")
table = sorted(result, key=lambda d: d['FDR'])[:10]
for row in table:
    logger.info('| {:4} | {:10} | {:^4} |'.format(row["classifier"], row["sampler"], row["FDR"]))
# RECALL
logger.info("RECALL..")
table = sorted(result, key=lambda d: d['RECALL'], reverse=True)[:10]
for row in table:
    logger.info('| {:4} | {:10} | {:^4} |'.format(row["classifier"], row["sampler"], row["RECALL"]))
# PRECISION
logger.info("PRECISION..")
table = sorted(result, key=lambda d: d['PRECISION'], reverse=True)[:10]
for row in table:
    logger.info('| {:4} | {:10} | {:^4} |'.format(row["classifier"], row["sampler"], row["PRECISION"]))






