"""
Light execution: 'python main.py --mode light --source dataset/data.csv'
Heavy execution: 'python main.py --mode heavy --source dataset/data.csv'
"""

import argparse
from library.Training.CrossValidation import CrossValidation


MODE_HELP = """'heavy' allows to reach the best performances regarding highlighted metrics inside README.md.
            HyperParam optimization is used and each model is evaluated 5 times in order to report average results.
            Expected execution time is around 40 min (AMD Ryzen 5)
    
            'light' is a lightweight version of the project that runs each only model once without
            executing any HyperParam optimization"""
SOURCE_HELP = "Expected path where dataset is located. Dataset is expected under csv format with well-known structure."
VERBOSE_HELP = """1 if it's necessary to print each milestone while executing the script.
               0 reports information inside log/ directory."""
STORE_NORMALIZED_HELP = """1 if the normalized dataset should be normalized, 0 otherwise"""

parser = argparse.ArgumentParser(description='German credit data ML model.')
parser.add_argument('--mode', type=str, default='light', choices=['light', 'heavy'], help=MODE_HELP)
parser.add_argument('--source', type=str, default='dataset/data.csv', help=SOURCE_HELP)
parser.add_argument('--verbose', type=int, default=0, choices=[0, 1], help=VERBOSE_HELP)
parser.add_argument('--store_normalized', type=int, default=1, choices=[0, 1], help=STORE_NORMALIZED_HELP)

args = parser.parse_args()

c = CrossValidation(args)
c.execute()
c.print_result()


