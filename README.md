### Telco Customer Churn - EDA & Classification ML Models 

The aim of this project is to create a machine learning model that predicts customer churn (stopping using company's service) based on informations of customers who left or stayed within the last month. 

The dataset used in this project is obtained from [Kaggle - Telco Customer Churn ](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data).

## Data preparation 

Used dataset has already been prepared and has no missing values and contains 17 categorical values a 3 continuous values. 

## Exploratory data analysis 

1. Search for the most correlated feature to our target which is "Churn" coulumn.
2. Make visualizations including previously specified features and "Churn" column in order to see pattern occuring in our data. 
3. Check if we are dealing with balanced or not classes in order to help ourselves with choosing ML model.

## Machine Learning Models 

1. Chosen ML models:
   - RandomForestClassifier
   - AdaBoostClassifier
   - GradientBoostingClassifier
Decision based on the fact that we are dealling with a bit imbalanced classes: 73,42 % rows in dataset have "No" in "Churn" column.

## Models evaluation

1. Classification report - great importance in our is attached to recall of "Yes" value as we want our model to misclassify as little as possible customers that want to stop using company's service.
2. Display confusion matrix.
3. Roc auc score.

## Results and conclusions




   

