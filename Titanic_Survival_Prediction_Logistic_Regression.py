#Import Dependencies for Running the Model
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Import Datasets
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Check Data Information for Train Data
df_train.dtypes
df_train.info()

# Check Data Information for Test Data
df_test.dtypes
df_test.info()

# Carry out Preprocessing on Train Data and  Drop Unneeded Columns
cla = {1:"First Class", 2: "Second Class", 3 :"Third Class"}
emb = {"Q" : "Queenstown" , "S" : "Southampton" , "C" : "Cherbourg" }
df_train['Pclass'] = df_train['Pclass'].map(cla)
df_train['Embarked'] = df_train['Embarked'].map(emb)

drop_cols = ["Name", "Cabin","Ticket"]
dummy_cols = ["Sex", "Pclass","Embarked"]

dummies_tr = []
for col in dummy_cols:
    dummies_tr.append(pd.get_dummies(df_train[col]))

df_dummy_tr = pd.concat(dummies_tr, axis = 1)
df_train = pd.concat((df_train, df_dummy_tr), axis =1)

df_train.drop(drop_cols, axis = 1, inplace = True)
df_train.drop(dummy_cols, axis=1, inplace=True)

df_train["Age"] = df_train["Age"].interpolate()
df_train["Age"] = df_train["Age"].astype('int64')

# Carry out Preprocessing on Test Data and Drop Unneeded Columns
cla = {1:"First Class", 2: "Second Class", 3 :"Third Class"}
emb = {"Q" : "Queenstown" , "S" : "Southampton" , "C" : "Cherbourg" }
df_test['Pclass'] = df_test['Pclass'].map(cla)
df_test['Embarked'] = df_test['Embarked'].map(emb)

drop_cols = ["Name", "Cabin","Ticket"]
dummy_cols = ["Sex", "Pclass","Embarked"]

dummies_te = []
for col in dummy_cols:
    dummies_te.append(pd.get_dummies(df_test[col]))

df_dummy_te = pd.concat(dummies_te, axis = 1)
df_test = pd.concat((df_test, df_dummy_te), axis =1)

df_test.drop(drop_cols, axis = 1, inplace = True)
df_test.drop(dummy_cols, axis=1, inplace=True)

df_test["Fare"] = df_test["Fare"].interpolate()
df_test["Age"] = df_test["Age"].interpolate()
df_test["Age"] = df_test["Age"].astype('int64')

# Split the Data into Features and Target Variables
# Train Data Feature and Target
X = df_train[['Age','SibSp','Parch','Fare','female','male', 'First Class', 'Second Class', 'Third Class', 'Cherbourg',
       'Queenstown', 'Southampton']]
y = df_train["Survived"]

# Test Data Features
X_test = df_test[['Age','SibSp','Parch','Fare','female','male', 'First Class', 'Second Class', 'Third Class', 'Cherbourg',
       'Queenstown', 'Southampton']]

#Instantiate Logistic Regression Model and Fit the Model on the Training Dataset
model = LogisticRegression(solver = "liblinear", max_iter = 1000, random_state = 42)
model.fit(X,y)

# Check Accuracy of the Model with the Training Data
train_acc = accuracy_score(y, model.predict(X))
print(train_acc)

# Predict Survival in the Test Dataset Using "model.predict" on the Test Features
predictions = model.predict(X_test)

# Print the Results and Export to CSV File
model_output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
model_output.to_csv('titanic_survivors.csv', index=False)
print("Your submission was successfully saved!")

# Check how many survivors/non-survivors the Model Predicted
model_output["Survived"].value_counts()

# Export the Model to a .py File Using the Pickle Module
pickle.dump(model, open('model.pkl', 'wb'))