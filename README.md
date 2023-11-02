# credit-risk-classification

## Analysis

### Overview

This repository contains two files: `credit_risk_classification.ipynb` and `lending_data.csv`. The credit_risk_classification notebook contains the code that has two Logistic Regression models which aim to predict the likelihood of a loan being healthy or at high risk of defaulting. The csv file contains the dataset used to train and test both of the models. Being able to predict the potential status of a loan (healthy or high-risk) is important for lending services companies to take into account as they approve or deny various loan applications. 

The `lending_data.csv` is historical lending activity data from the company, compiled in such a way that some of the column headers are: loan_size, interest_rate, derogatory_marks, total_debt, and more. Across the whole csv file, we have 77,536 rows of data to work with. 75,036 of these are classified as healthy loans and 2,500 of which are classified as high risk loans. We want to use this historical data to train and test a Logistic Regression model to be used on new loan applicants to attempt to predict if that loan will be classified as healthy or high risk before the lending servicer approves the loan.

The two Logistic Regression models were created in largely the same way, with one seemingly small but significant difference. Model #1 is created using the original dataset, with the y variable consisting of the loan_status dataframe column, and the x variabale consisting of every other column in the dataframe. These x and y variables were then futher split into training and testing datasets, which by default is 75% of the data designated to the training process and 25% designated to the testing process. A Logistic Regression model was created using SKLearn, and then fit to the x_train and y_train data. Predictions were then made using this model and feeding it the x_test data. Model #2 shares these same characteristics, however before we could create the Logistic Regression model, a RandomOverSampler (ros) model was created and fit to the x_train and y_train data. The ros model works by randomly duplicating values in the minority class so that data in both the majority and minority classes are equally represented. Once the ros model was fit to the training data, we can now put these values into the Logistic Regression model and fit this model as well. Predictions were made using the Logistic Regression model and feeding it the x_test data.

### Machine Learning Model #1

- The accuracy score represents how often the model is correct, calculated by the ratio of correctly predicted observations to the total number of observations. The balanced_accuracy_score is used when dealing with imbalanced data. For this model, that score is 0.994 or 99.4%.
- The precision score is the percentage of correct positive predictions made by the model relative to the total number of positive predictions made. This model has a 100% precision score for healthy accounts (0) and 87% for high risk accounts (1).
- The recall score is the percentage of correct positive predictions made by the model relative to the total number of actual positive values in the dataset. This model has a 100% recall score for healthy accounts (0) and 89% for high risk accounts (1).

### Machine Learning Model #2

- The accuracy score represents how often the model is correct, calculated by the ratio of correctly predicted observations to the total number of observations. The balanced_accuracy_score is used when dealing with imbalanced data. For this model, that score is 0.996 or 99.6%
- The precision score is the percentage of correct positive predictions made by the model relative to the total number of positive predictions made. This model has a 100% precision score for healthy accounts (0) and 87% for high risk accounts (1).
- rThe recall score is the percentage of correct positive predictions made by the model relative to the total number of actual positive values in the dataset. This model has a 100% recall score for healthy accounts (0) and 100% for high risk accounts (1).

### Summary

Based purely on the accuracy score of both models shown above, it appears that Model #2 would be the top performer for generally predicting if a potential account would be at high risk of defaulting or not due to the accuracy percentage being 99.6%. However, there are many factors to consider alongside the accuracy score. A lending company might be most concerned with accurately predicting the true negative values, such as a healthy account (0) actually being healthy, or predicting the true positive values, such as a high risk account (1) actually ending in loan default. The insights a company chooses to priortize could potentially influence which of the two models is best suited to their unique needs, should one model outperform another in a specific metric. For this dataset and in either case, Model #2 still proves to perform the best.

I would need to discuss further needs of the company before being able to recommend one of the above models, but I am inclined to say it is likely that the best model for many situations is Model #2 that includes the random over sampling of minority data.


## References

No outside references were needed for the code portion of this repository. The following links were referenced to further my understanding of the various model scoring metrics which was then used in the above analysis:

https://www.statology.org/sklearn-classification-report/

https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
