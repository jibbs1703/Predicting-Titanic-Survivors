# Using-ML-for-Predicting-Survivors---The-Case-of-the-Titanic-Disaster
This repository contains model(s) for predicting survivors in the titanic disaster of 1912. This project is a submission to the Kaggle "Getting Started with Prediction" Competition. As part of the competition, competitors are required to use machine learning to create a model that predicts which passengers survived the Titanic shipwreck. The resulting prediction models are scored based on their accuracy. However, this project the most basic classification model - the Logistic regression Model, in its prediction of the shipwreck survivors.

## Dataset For Analysis
The dataset used for this ML project was obtained from [Kaggle](https://www.kaggle.com/competitions/titanic) and has been split into test and train sections. The train data (train.csv) which contains 891 observations is used to model the predictions of survival and subsequently used on the test data (test.csv) which contains 418 observations. The titanic data contains passenger information like Name, Sex, Age, Passenger Class (1st, 2nd or 3rd), Number of Siblings/Spouses Aboard, Number of Parents/Children Aboard, Ticket Number, Passenger Fare (in British pound), 
Cabin, and the Port of Embarkation (Cherbourg, Queenstown or Southampton). Based on these features, the passengers either survive or not surviive after the shipwreck. 

## Model Used
- Logistic Regression Model

## Python Packages Needed
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-Learn](https://scikit-learn.org/)
  
## Model Results
With a training accuracy of 80%, the trained model predicted 36% of the passengers in the Test Dataset survived the shipwreck while 64% did not survive. The Kaggle submission for this model returned a 76.94% accuracy score, indicating that there is room for improvement on the Logistic Regression model used to make the predictions. This project uses the most basic classification model and further projects hope to use more models in making predictions/classifications.

## Author(s)
- **Abraham Ajibade** [Linkedin](https://www.linkedin.com/in/abraham-ajibade-759772117)
- **Boluwtife Olayinka** [Linkedin](https://www.linkedin.com/in/ajibade-bolu/)

## Code Use and Re-use
This repository and the code therein can be used by everyone, with full permission from the authors. 
