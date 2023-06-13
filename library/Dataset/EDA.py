"""
Normalization rules

- WARNING: 0 values
- No Null values

# Attribute 1: status of existing checking account
    A11 :      ... <    0 DM - 1
    A12 : 0 <= ... <  200 DM - 2
    A13 :      ... >= 200 DM - 3
    A14 : no checking account - 4

    DM -> Deutsche Mark

# Attribute 3: credit history
    A30 : no credits taken / all credits paid back duly - 0
    A31 : all credits at this bank paid back duly - 1
    A32 : existing credits paid back duly till now - 2
    A33 : delay in paying off in the past - 3
    A34 : critical account / other credits existing (not at this bank) - 4

# Attribute 4:  Purpose
    A40 : car (new) - 0
    A41 : car (used) - 1
    A42 : furniture/equipment - 2
    A43 : radio/television - 3
    A44 : domestic appliances - 4
    A45 : repairs - 5
    A46 : education - 6
    A47 : (vacation - does not exist?) - 7
    A48 : retraining - 8
    A49 : business - 9
    A410 : others - 10

# Attribute 6: savings account/bonds
    A61 :          ... <  100 DM - 1
    A62 :   100 <= ... <  500 DM - 2
    A63 :   500 <= ... < 1000 DM - 3
    A64 :          .. >= 1000 DM - 4
    A65 :   unknown/ no savings account - 5

# Attribute 7: present employment since
    A71 : unemployed - 1
    A72 :       ... < 1 year - 2
    A73 : 1  <= ... < 4 years - 3
    A74 : 4  <= ... < 7 years - 4
    A75 :       .. >= 7 years - 5

# Attribute 9: personal status and sex
    A91 : male   : divorced/separated - 1
    A92 : female : divorced/separated/married - 2
    A93 : male   : single - 3
    A94 : male   : married/widowed - 4
    A95 : female : single - 5

# Attribute 10: other debtors / guarantors
    A101 : none - 1
    A102 : co-applicant - 2
    A103 : guarantor - 3

# Attribute 12: property
    A121 : real estate - 1
    A122 : if not A121 : building society savings agreement/life insurance - 2
    A123 : if not A121/A122 : car or other, not in attribute 6 - 3
    A124 : unknown/no property - 4

Attribute 14: other installment plans
    A141 : bank - 1
    A142 : stores - 2
    A143 : none - 3

Attribute 15: housing
    A151 : rent - 1
    A152 : own - 2
    A153 : for free - 3

Attribute 17: job
    A171 : unemployed/unskilled/non-resident - 1
    A172 : unskilled/resident - 2
    A173 : skilled employee/official - 3
    A174 : management/self-employed/highly qualified employee/officer - 4

Attribute 19: telephone
    A191 : none - 1
    A192 : yes, registered under the customers name - 2

Attribute 20: foreign worker
    A201 : yes - 1
    A202 : no - 2

"""
import pandas as pd, json
from sklearn.preprocessing import StandardScaler
from library.Exceptions.CustomExceptions import NormalizationException
from library.Dataset.Dataset import Dataset


class EDA:
    """
    EDA class executes data transformation based on above rules.
    Scaling and null-values research is also applied.
    Scaling must be applied ONLY 4 training set.
    """

    def __init__(self, df: pd.DataFrame, verbose=0, path_to_save='dataset/data_normalized.{}'):
        """
        :param df: DataFrame to normalize
        """
        self.df = df
        self.verbose = verbose
        self.base_path = "library/Dataset/Rules/{}"
        self.path_to_save = path_to_save

    def get_df(self):
        return self.df

    def null_val_replace(self, save=False):
        """
        Replace null values with median value
        :param save: var. used to save or not new normalized DataFrame
        :return: nothing
        """
        try:
            null_val = int(self.df.isna().sum().sum())
            if self.verbose: print(f"Null values found: {null_val}")

            if null_val > 0:
                for key in self.df.keys():
                    self.df[key] = self.df[key].fillna(self.df[key].median())

                if self.verbose: print("Null values replaced...")

            ds = Dataset()
            if save:
                ds.store_dataframe(self.df, path=self.path_to_save)

        except Exception as e:
            raise Exception(f'Error while replacing null values: {e}')

    @staticmethod
    def scaling(x_training):
        """
        Training set is scaled using StandardScaler.
        :param x_training: matrix to be normalized
        :return: x_training normalized
        """
        scaler = StandardScaler()
        x_training = scaler.fit_transform(x_training)
        return x_training

    def replacing(self, style="DEFAULT", save=False):
        """
        Execute rules transformation with factory builder design pattern
        :param style: var. used to choose between different rules
        :param save: var. used to save or not new normalized DataFrame
        :return:
        """
        if style == "DEFAULT":
            self.__rules_default(save=save)
        else:
            raise NormalizationException(f"Transformation rules '{style}' not defined")

    def __rules_default(self, save=False):
        """
        This method apply rules to library attribute self.df.
        self.df will be stored inside a new CSV / XLSX file.

        First normalize data then convert 'em all to integer.
        :param save: var. used to save or not new normalized DataFrame
        :return: nothing
        """
        path = self.base_path.format("default.json")
        f = open(path)
        replacements = json.load(f)
        f.close()

        # normalization with replacements rules & cast to integer

        for col in replacements:
            self.df = self.df.replace(to_replace=replacements[col])
            self.df = self.df.astype({col: 'int'})

        ds = Dataset()
        if save:
            ds.store_dataframe(self.df, path=self.path_to_save)
