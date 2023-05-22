# StatLog - German Credit Data
This ML poject estimates is a customer is a good or bad borrower. It estimates the reliability of a potential customer under a binary classification model.
## Description
The dataset was donated in 1994. It's made of 20 attributes 13 categorical and 7 numerical. There are 1000 samples.
Ref.: https://archive-beta.ics.uci.edu/dataset/144/statlog+german+credit+data
## ML Model
The ML model used is the Random Forest and it's trained using under-sample techniques due to class unbalance and different class weight.
### Data description
Features:

    - Attribute 1: Status of existing checking account (qualitative)
               
               A11 :      ... <    0 DM
               A12 : 0 <= ... <  200 DM
               A13 :      ... >= 200 DM / salary assignments for at least 1 year
               A14 : no checking account

    - Attribute 2: Duration in month (numerical)
       

    - Attribute 3: Credit history (qualitative)
              
              A30 : no credits taken / all credits paid back duly
              A31 : all credits at this bank paid back duly
              A32 : existing credits paid back duly till now
              A33 : delay in paying off in the past
              A34 : critical account / other credits existing (not at this bank)

    - Attribute 4: Purpose (qualitative)
              
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

    - Attribute 5: Credit amount (numerical)
              

    - Attribute 6: Savings account/bonds (qualitative)
              
              A61 :          ... <  100 DM
              A62 :   100 <= ... <  500 DM
              A63 :   500 <= ... < 1000 DM
              A64 :          .. >= 1000 DM
              A65 :   unknown / no savings account

    - Attribute 7: Present employment since (qualitative)
              
              A71 : unemployed
              A72 :       ... < 1 year
              A73 : 1  <= ... < 4 years
              A74 : 4  <= ... < 7 years
              A75 :       .. >= 7 years

    - Attribute 8: Installment rate in percentage of disposable income (numerical)
              

    - Attribute 9: Personal status and sex (qualitative)
              
              A91 : male   : divorced / separated
              A92 : female : divorced / separated / married
              A93 : male   : single
              A94 : male   : married / widowed
              A95 : female : single

    - Attribute 10: Other debtors / guarantors (qualitative)
              
              A101 : none
              A102 : co-applicant
              A103 : guarantor

    - Attribute 11: Present residence since (numerical)
              

    - Attribute 12: Property (qualitative)
              
              A121 : real estate
              A122 : if not A121 : building society savings agreement / life insurance
              A123 : if not A121/A122 : car or other, not in attribute 6
              A124 : unknown / no property

    - Attribute 13: Age in years (numerical)
              

    - Attribute 14: Other installment plans (qualitative)
              
              A141 : bank
              A142 : stores
              A143 : none

    - Attribute 15: (qualitative)
              Housing
              A151 : rent
              A152 : own
              A153 : for free

    - Attribute 16: Number of existing credits at this bank (numerical)
                  

    - Attribute 17: Job (qualitative)
              
              A171 : unemployed / unskilled  - non-resident
              A172 : unskilled - resident
              A173 : skilled employee / official
              A174 : management / self-employed / highly qualified employee/ officer

    - Attribute 18:  Number of people being liable to provide maintenance for (numerical)
             

    - Attribute 19: Telephone (qualitative)
              
              A191 : none
              A192 : yes, registered under the customers name

    - Attribute 20: Foreign worker (qualitative)
              
              A201 : yes
              A202 : no
Evaluation:

    - This dataset requires use of a cost matrix:


          |  1   |   2  |
        ----------------|
        1 |  0   |   1  |
        ----------------|
        2 |  5   |   0  |

        (1 = Good,  2 = Bad)

        the rows represent the actual classification and the columns
        the predicted classification.

        It is worse to class a customer as good when they are bad (5),
        than it is to class a customer as bad when they are good (1).

        It's actually better to have a lower FP reather than a lower FN
        We prefer precision to recall 'cause we want to minimize the FP.


### Algorithm
The used algorithm is Random Forest which belongs to Ensemble algorithm family. The main focus is based on the dataset.</br> 
Data are unbalanced, 'Bad borrower' class which is strictly required to be more accurate than the first one ('Good borrower') is less represented.</br>
In order to balance the dataset it's used the SVM SMOTE technique which oversample the less represented class.</br>
Categorical variables are actually classified with integer values under the Replacing technique. The corresponding value for a categorical attribute is described as: <br />       ```
        cat = A201 -> int(cat[-2:])
    ```
Logical Regression model is also implemented but not suggested due to its high FDR.</br>
Oversampling K-SMOTE, ADASYN and undersample techniques are also implemented but not suggested. K-SMOTE has higher PRECISION and lower FDR but it's not used 'cause it's to poor with TrueNegative detection (poor Recall).<br>
ADASYN and SVM are pretty similar but SVM is slightly better for FDR and F1-Score.</br>

Current F1-SCORE: 84%
Current Precision: 87%
Current FDR: 13%
Current Recall/Sensitivity: 82%
Current Classifier: Random forrest
Oversampling technique:: SVM SMOTE

Confusion Matrix images are linked at: [graph comparison](/classifier/doc)
Extra documentation at: [document](/FdML_Report_Template__1_.pdf)
## Web View
There i a website to test out the ML model capabilities.
Ref.: [https://apanseratesting.pythonanywhere.com/page/stat_log](http://apanseratesting.pythonanywhere.com/page/stat_log)


