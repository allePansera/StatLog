"""
    Attribute 1:  (qualitative)
               Status of existing checking account
               A11 :      ... <    0 DM
               A12 : 0 <= ... <  200 DM
               A13 :      ... >= 200 DM /
                 salary assignments for at least 1 year
               A14 : no checking account

    Attribute 2:  (numerical)
              Duration in month

    Attribute 3:  (qualitative)
              Credit history
              A30 : no credits taken/
                all credits paid back duly
              A31 : all credits at this bank paid back duly
              A32 : existing credits paid back duly till now
              A33 : delay in paying off in the past
              A34 : critical account/
                other credits existing (not at this bank)

    Attribute 4:  (qualitative)
              Purpose
              A40 : car (new)
              A41 : car (used)
              A42 : furniture/equipment
              A43 : radio/television
              A44 : domestic appliances
              A45 : repairs
              A46 : education
              A47 : (vacation - does not exist?)
              A48 : retraining
              A49 : business
              A410 : others

    Attribute 5:  (numerical)
              Credit amount

    Attibute 6:  (qualitative)
              Savings account/bonds
              A61 :          ... <  100 DM
              A62 :   100 <= ... <  500 DM
              A63 :   500 <= ... < 1000 DM
              A64 :          .. >= 1000 DM
              A65 :   unknown/ no savings account

    Attribute 7:  (qualitative)
              Present employment since
              A71 : unemployed
              A72 :       ... < 1 year
              A73 : 1  <= ... < 4 years
              A74 : 4  <= ... < 7 years
              A75 :       .. >= 7 years

    Attribute 8:  (numerical)
              Installment rate in percentage of disposable income

    Attribute 9:  (qualitative)
              Personal status and sex
              A91 : male   : divorced/separated
              A92 : female : divorced/separated/married
              A93 : male   : single
              A94 : male   : married/widowed
              A95 : female : single

    Attribute 10: (qualitative)
              Other debtors / guarantors
              A101 : none
              A102 : co-applicant
              A103 : guarantor

    Attribute 11: (numerical)
              Present residence since

    Attribute 12: (qualitative)
              Property
              A121 : real estate
              A122 : if not A121 : building society savings agreement/
                       life insurance
              A123 : if not A121/A122 : car or other, not in attribute 6
              A124 : unknown / no property

    Attribute 13: (numerical)
              Age in years

    Attribute 14: (qualitative)
              Other installment plans
              A141 : bank
              A142 : stores
              A143 : none

    Attribute 15: (qualitative)
              Housing
              A151 : rent
              A152 : own
              A153 : for free

    Attribute 16: (numerical)
                  Number of existing credits at this bank

    Attribute 17: (qualitative)
              Job
              A171 : unemployed/ unskilled  - non-resident
              A172 : unskilled - resident
              A173 : skilled employee / official
              A174 : management/ self-employed/
                 highly qualified employee/ officer

    Attribute 18: (numerical)
              Number of people being liable to provide maintenance for

    Attribute 19: (qualitative)
              Telephone
              A191 : none
              A192 : yes, registered under the customers name

    Attribute 20: (qualitative)
              foreign worker
              A201 : yes
              A202 : no

"""
import pandas as pd
import numpy as np
import json, joblib
from datetime import datetime
from library.Dataset.Normalization import Normalization


class TestCustomer:
    def __init__(self):
        """
        Constructor loads default features to be asked.
        """
        self.default_path = "library/Dataset/Rules/default.json"
        self.classifier_path = "classifier/rf.joblib"
        self.history_path = "history/input_{}.json"
        self.classifier = joblib.load(self.classifier_path)
        f = open(self.default_path)
        self.values = json.load(f)
        f.close()

    def store_df(self, df: pd.DataFrame):
        """
        This method is used to store a 'handmade' DataFrame.
        store_df() allows to dynamically retrain the ML model.
        :param df: DataFrame row
        :return: Nothing
        """
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = self.history_path.format(timestamp)
        f = open(filename, "w")
        df.to_json(f)
        f.close()

    def ask_4_values(self):
        """
        This method is used to ask feature value to the customer 4 tests
        :return: answer structured in order to be read from predict() method. DataFrame is normalized.
        """
        answer = {}
        for param in self.values:
            if param != 'Target':
                answer[param] = input(f'Enter {param}:\n')

        return self.normalize(answer)

    def normalize(self, answer_dict: dict):
        """
        Given a dict this method normalize it and produce the df used from ml model
        :param answer_dict: answer dict {key:value}
        :return: DataFrame normalized
        """
        answer = {}
        for param in self.values:
            answer[param] = answer_dict.get(param, 0)

        series = np.fromiter(answer.values(), dtype='object')
        columns = answer.keys()
        df = pd.DataFrame(data=[series], columns=columns)
        normalizer = Normalization(df)
        df = normalizer.execute()
        df.pop('Target')
        return df

    def predict(self, feature_values: pd.DataFrame):
        """
        This method is used to predict customer reliability.
        :param feature_values: it's the input values used to predict customer trust
        :return: integer value: 1 - Good borrower; 2 - Bad borrower
        """
        y_predicted = self.classifier.predict(feature_values)
        return y_predicted

