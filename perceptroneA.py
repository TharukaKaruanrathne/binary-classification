


# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:17:34 2023

@author: Niluksha
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix

# Load the student dataset
student_dataset = pd.read_csv("student.txt")

student_dataset = pd.read_csv("student.txt")

# Rename the columns
student_dataset.columns = ['index',	'age','failures','Dailyalc','Weekendalc',	'AvgofAlc_consumption', 'health'	
] 

#print(student_dataset)

#print(student_dataset.describe().transpose())

#print(student_dataset.shape)

x = student_dataset.drop('AvgofAlc_consumption', axis =1)
y = (student_dataset['AvgofAlc_consumption'] >= 3).astype(int)

# Split the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y)

#scale features
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Create and train the Perceptron model
perceptron = Perceptron(max_iter=500)
perceptron.fit(x_train, y_train)

# Make predictions on the testing set
predictions = perceptron.predict(x_test)

# Evaluate the model

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

