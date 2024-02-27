# Using-ML-for-Predicting-Survivors---The-Case-of-the-Titanic-Disaster
This repository contains model(s) for predicting survivors in the titanic disaster of 1912. This project is a submission to the Kaggle "Getting Started with Prediction" Competition. As part of the competition, competitors are required to use machine learning to create a model that predicts which passengers survived the Titanic shipwreck. The resulting prediction models are scored based on their accuracy. However, this project uses a few model types and checks for the accuracy scores across the board.

## Model(s) Used
- XGBoost
- Random Forest Classifier
- Gradient Boosting
- Logistic Rrgression

## Dataset For Analysis
The dataset used for this ML project was obtained from [Kaggle](https://www.kaggle.com/competitions/titanic) and has been split into test and train sections. The train data (train.csv) which contains 891 observations is used to model the predictions of survival and subsequently used on the test data (test.csv) which contains 418 observations. The titanic data contains passenger information like name, age, gender, socio-economic class, etc. Using these features, the model predicts a 0 or 1 value for every observation, 0 represents non-survival, while 1 represents survival after the shipwreck. 

## Python Packages Needed
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-Learn](https://scikit-learn.org/)

## Author(s)
- **Abraham Ajibade** [Linkedin](https://www.linkedin.com/in/abraham-ajibade-759772117)
- **Boluwtife Olayinka** [Linkedin](https://www.linkedin.com/in/ajibade-bolu/)
