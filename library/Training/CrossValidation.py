from library.Training.Training import Training
from library.Dataset.DatasetPartition import DatasetPartition
from library.Dataset.Dataset import Dataset
from library.Dataset.Normalization import Normalization

# Logger config

# Const definition
SUPPORTED_CLASSIFIER = ["RF", "LR"]
SUPPORTED_SAMPLING = ["US", "OS_SVM", "OS_K", "OS_ADASYN"]
K_TEST = 10
LOG_BASE_PATH = "log/testing{}_{}.log"
performances = {}


## Dataset division
# Dataset download
ds = Dataset()
df = ds.download(save=True)
# Dataset normalization
normalizer = Normalization(df)
df = normalizer.execute(save=True)
# Dataset partition
dp = DatasetPartition(df)
x_training, y_training, x_testing, y_testing = dp.split()

## Model-selection
for classifier in SUPPORTED_CLASSIFIER:
    for sampler in SUPPORTED_SAMPLING:
        performances[classifier][sampler] = []
        for index in range(K_TEST):
            t = Training(method="RF", oversample_tech="OS_SVM", logging_path='log/training.log')
            f1, fdr, precision, recall = t.run()
            test_result = {"F1": f1, "FDR": fdr, "PRECISION": precision, "RECALL": recall}
            performances[classifier][sampler].append(test_result)




