#Develop and train our model

#Data location - https://github.com/vyashemang/flask-salary-predictor/blob/master/Salary_Data.csv

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json

#We will predict the salary of an employee based on his/her experience in the field
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting our data into train and test size of 0.67 and 0.33 respectively using train_test_split from sklearn.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


#The object is instantiated as a regressor of the class LinearRegression() 
# and trained using X_train and y_train. 
# Latter the predicted results are stored in the y_pred.
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


#We will save our trained model to the disk using the pickle library. 
# Pickle is used to serializing and de-serializing a Python object structure. 
# In which python object is converted into the byte stream. 
# dump() method dumps the object into the file specified in the arguments.
pickle.dump(regressor, open('model.pkl','wb'))

#pickle.load() method loads the method and saves the deserialized bytes to model. 
# Predictions can be done using model.predict().
model = pickle.load(open('model.pkl','rb'))

#predict the salary of the employee who has experience of 1.8 years.
print(model.predict([[1.8]]))

