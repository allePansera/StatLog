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
import pandas as pd
from library.Exceptions.CustomExceptions import NormalizationException


class Normalization:
    """
    Normalization class executes data transformation based on above rules
    """

    def __init__(self, df: pd.DataFrame):
        """
        :param df: DataFrame to normalize
        """
        self.df = df

    def execute(self, style="DEFAULT"):
        """
        Execute rules transformation with factory builder design pattern
        :param style: var. used to choose between different rules
        :return: updated DataFrame
        """
        if style == "DEFAULT":
            self.rules_default()
            return self.df
        else:
            raise NormalizationException(f"Transformation rules '{style}' not defined")

    def rules_default(self):
        """
        This method apply rules to library attribute self.df
        :return: nothing
        """
        pass
