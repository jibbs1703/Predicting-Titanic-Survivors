# Predicting-Titanic-Survivors
This repository contains a model for predicting the survivors of the titanic disaster of 1912. The project results were submitted to the Kaggle "Getting Started with Prediction" Competition. As part of the competition, competitors are required to use machine learning model(s) of their choice in predicting which passengers survived the Titanic shipwreck of 1912. The resulting prediction models are then scored, based on their accuracy.

## Dataset For Analysis
The dataset used for this ML project was obtained from [Kaggle](https://www.kaggle.com/competitions/titanic) and has been split into test and train sections. The train data (train.csv) which contains 891 observations is used to model the predictions of survival and subsequently used on the test data (test.csv) which contains 418 observations. The titanic data contains passenger information like Name, Sex, Age, Passenger Class (1st, 2nd or 3rd), Number of Siblings/Spouses Aboard, Number of Parents/Children Aboard, Ticket Number, Passenger Fare (in British pound), Cabin, and the Port of Embarkation (Cherbourg, Queenstown or Southampton). Based on these features, the passengers either survive or not surviive after the shipwreck. 

## Python Packages and Modules Needed
It is worthy to note that the train-test-split method does not need to be imported from Scikit-Learn because the data has been split prior to the start of the task. However, to make the cleaning and feature engineering a blanket process, the train and test datasets are merged and later split using the train-test-split method (I think this is an easier alternative to working on the splits separarely or using a function to handle the cleaning and feature engineering of both datasets). 

[Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) are imported for Data Manipulation and Wrangling. The Machine Learning Models are imported from [Scikit-Learn](https://scikit-learn.org/). Each model's prediction is then saved as a .pkl file using the Pickle module. 

## Chronology 
- Import necessary libraries and datasets.
- Identify datatypes and make appropriate conversions.
- Preprocess dataset - deal with missing values and labels in numeric and categorical features respectively.
- Split data into the features and target.
- Training machine learning models on training dataset and check training accuracy. 
- Use trained model on test dataset and make prediction.
- Save predictions to a desired file format file (in this case, a comma separated value - csv file). 
- Dump the model into a pickle file and 

## Model Results
**Logistic Regression Model** : The Logistic Regression Model yielded an accuracy of 80% with used on the test data, predicting that 36% of the passengers in the Test Dataset survived the shipwreck while 64% did not survive. The Kaggle submission for this model returned a 76.94% accuracy score. 

**Logistic Regression Cross Validation Model** : The Logistic Regression Cross Validation Model yielded an accuracy of 81% with used on the test data, predicting that 37% of the passengers in the Test Dataset survived the shipwreck while 63% did not survive. The Kaggle submission for this model returned a 77.51% accuracy score.

## Author(s)
- [**Abraham Ajibade**](https://www.linkedin.com/in/abraham-ajibade-759772117)
- [**Boluwtife Olayinka**](https://www.linkedin.com/in/ajibade-bolu/)
