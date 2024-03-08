# Predicting-Titanic-Survivors-Using-the-Logistic-Regression-Model
This repository contains a model for predicting the survivors of the titanic disaster of 1912. This project is a submission to the Kaggle "Getting Started with Prediction" Competition. As part of the competition, competitors are required to use a machine learning model of their choice in predicting which passengers survived the Titanic shipwreck of 1912. The resulting prediction models are then scored, based on their accuracy. However, this project the most basic classification model - the Logistic regression Model, in its prediction of the shipwreck survivors.

## Dataset For Analysis
The dataset used for this ML project was obtained from [Kaggle](https://www.kaggle.com/competitions/titanic) and has been split into test and train sections. The train data (train.csv) which contains 891 observations is used to model the predictions of survival and subsequently used on the test data (test.csv) which contains 418 observations. The titanic data contains passenger information like Name, Sex, Age, Passenger Class (1st, 2nd or 3rd), Number of Siblings/Spouses Aboard, Number of Parents/Children Aboard, Ticket Number, Passenger Fare (in British pound), 
Cabin, and the Port of Embarkation (Cherbourg, Queenstown or Southampton). Based on these features, the passengers either survive or not surviive after the shipwreck. 

## Python Packages Needed
It is worthy to note that the train-test-split imported from skicit-learn is not needed because the data has been split prior to the start of the task. and should be used if data comes without a train-test split. Pandas and Numpy are imported for Data Manipulation and Wrangling while the Logistic Regression is imoprted from Skicit-Learn.

- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-Learn](https://scikit-learn.org/)

## Model Description and Steps in Code Script
The Logistic Regression Model was used to predict the survivors of the shipwreck. Logistic regression is a statistical method used for binary classification, i.e., predicting the probability of an observation belonging to one of two possible classes. In the case of the Titanic Shipwreck, the model predicts the probability of a passenger either surviving or not. The model was chosen for this project due to its simplicity and its interpretability. 

The code script shows the steps taken in training a Logistic Regression Model for making predictions of survival for passengers in the Titanic Shipwreck of 1912.The script shows how to do exploratory data analysis (EDA) and pre-processing the data for the model. It shows how to identify datatypes, make appropriate conversions, drop features that have excessive missing values and for those observations with few missing values, it shows how impute values for the missing observations called "NaNs." The code script also shows how to create dummies for categorical variables in the dataset while also dropping the original columns.

Finally, it also shows how to train a Logistic Regression Model using a training dataset, how to use the trained model on test datasets, how to make predictions on the test dataset and save these predictions to a desired file format file (in this case, a comma separated value (csv) file).

## Model Results
With a training accuracy of 80%, the trained model predicted 36% of the passengers in the Test Dataset survived the shipwreck while 64% did not survive. The Kaggle submission for this model returned a 76.94% accuracy score, indicating that there is room for improvement on the Logistic Regression model used to make the predictions. This project uses the most basic classification model and further projects hope to use more sophisticated models in making predictions/classifications.

## Author(s)
- **Abraham Ajibade** [Linkedin](https://www.linkedin.com/in/abraham-ajibade-759772117)
- **Boluwtife Olayinka** [Linkedin](https://www.linkedin.com/in/ajibade-bolu/)

## Code Use and Re-use
This repository and the code therein can be used by everyone, with full permission from the authors. 
