# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:59:52 2025

@author: Swapnil Mishra
"""

'''CRISP-ML(Q):
Business Problem: 
There are a lot of assumptions in the diagnosis pertaining to cancer. In a few cases radiologists,
pathologists and oncologists fo wrong in diagnosing whether tumor is benign (non cancerous ) or malignant (cancerous).
Hence team of physicians want us to build an AI application which will predict with confidence the presence of cancer 
in a patient.This will serve as a compliment to physicians.

Business Objective : Maximize cancer detection.
Business Constrains: Minimize treatment cost & Maximize patient convenience.

Success Criteria :
Business Seccess Criteria : Increase the correct diagnosis of cancer in at least 96% of patients.
Machine learning Success criteria : Achieve an accuracy of atleast 98%
Economic success criteria : Reducing medical expenses will improve trust of patients and thereby hospital will see an increase in revenue by atleast 12%   

Data Collection: 
Data is collected from the hospital for 569 patients,30 features and 1 label comprise the feature set.
Ten real-valued features are computed for each cell nucleus:
    
a) radius(mean of distances from cancer to points on the perimeter)
b) texture (standard deviation of grey-scale values)
c) perimeter
d) area 
e) smoothness(local variation in radius length)
f) compctness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation"-1)'''
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import sklearn.metrics as skmet
import joblib
import pickle

# PostgreSQL
# pip install psycopg2

# Psycopg2 is a PostgreSQL database driver, it is used to perform operations on PostgreSQL using Python, it is designed for multi-threaded applications

import psycopg2 
from sqlalchemy import create_engine

wbcd = pd.read_csv(r"C:\Users\Swapnil Mishra\Desktop\DS\KNN\KNN flask\wbcd.csv")

# Creating engine which connect to postgreSQL
# conn_string = psycopg2.connect(database = "postgres", user = 'postgres', password = 'monish1234', host = 'localhost', port= '5432')

conn_string = ("postgresql+psycopg2://{user}:{pw}@localhost/{db}"
                       .format(user = "postgres", # user
                               pw = "swap", # password
                               db = "wbcd_db")) # database

db = create_engine(conn_string)
conn = db.connect()

wbcd.to_sql('wbcd', con = conn, if_exists = 'replace', index = False)

conn.autocommit = True

# Select query
sql = 'SELECT * from wbcd'
wbcd_data = pd.read_sql_query(sql, conn)

wbcd_data

# converting B to Benign and M to Malignant 
wbcd_data['diagnosis'] = np.where(wbcd_data['diagnosis'] == 'B', 'Benign ', wbcd_data['diagnosis'])
wbcd_data['diagnosis'] = np.where(wbcd_data['diagnosis'] == 'M', 'Malignant ', wbcd_data['diagnosis'])

wbcd_data.drop(['id'], axis = 1, inplace = True) # Excluding id column

wbcd_data.info()   # No missing values observed

wbcd_data.describe()

wbcd_data.info()

numeric_features = wbcd_data.select_dtypes(exclude = ['object']).columns

numeric_features

num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'mean'))])

# Encoding on Sex column
categorical_features = ['Sex']

categorical_features

# Convert Categorical data "Sex" to Numerical data using OneHotEncoder

# DataFrameMapper is used to map the given Attribute

categ_pipeline = Pipeline([('label', DataFrameMapper([(categorical_features, OneHotEncoder(drop='if_binary'))]))])

preprocess_pipeline = ColumnTransformer([('categorical', categ_pipeline, categorical_features), 
                                       ('numerical', num_pipeline, numeric_features)])

processed = preprocess_pipeline.fit(wbcd_data)  # Pass the raw data through pipeline

processed

joblib.dump(processed, 'processed1')

import os 
os.getcwd()

wbcd = pd.DataFrame(processed.transform(wbcd_data))  # Clean and processed data for Clustering

wbcd

wbcd.columns

wbcd.info()

new_features = wbcd.select_dtypes(exclude=['object']).columns  
# Capture only numeric data. If in newcase we have any non-numeric columns, we can skip them through.

new_features

scale_pipeline = Pipeline([('scale', MinMaxScaler())])

preprocess_pipeline2 = ColumnTransformer([('scale', scale_pipeline, new_features)], 
                                         remainder = 'passthrough') # Skips the transformations for remaining columns

processed2 = preprocess_pipeline2.fit(wbcd)

joblib.dump(processed2, 'processed2')

import os 
os.getcwd()

# Normalized data frame (considering the numerical part of data)

wbcd_n = pd.DataFrame(processed2.transform(wbcd))

wbcd_n.describe()

# Separating the input and output from the dataset
X = np.array(wbcd_n.iloc[:, :]) # Predictors 
Y = np.array(wbcd_data['diagnosis']) # Target

X

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

X_train.shape

X_test.shape

## K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 21)

KNN = knn.fit(X_train, Y_train)  # Train the kNN model

### Evaluate the model
# Evaluate the model with train data

pred_train = knn.predict(X_train)  # Predict on train data

pred_train

# Cross table
pd.crosstab(Y_train, pred_train, rownames = ['Actual'], colnames = ['Predictions']) 

print(skmet.accuracy_score(Y_train, pred_train))  # Accuracy measure

# Predict the class on test data
pred = knn.predict(X_test)
pred

# Evaluate the model with test data

print(skmet.accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 

cm = skmet.confusion_matrix(Y_test, pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Benign', 'Malignant'])
cmplot.plot()
cmplot.ax_.set(title = 'Cancer Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])
    
acc

# Train data accuracy plot 
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")

# Test data accuracy plot
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")

# Plotting the data accuracies in a single plot

plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")

plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")

from sklearn.model_selection import GridSearchCV
help(GridSearchCV)

k_range = list(range(3, 50, 2))
param_grid = dict(n_neighbors = k_range)
  
# Defining parameter range
grid = GridSearchCV(KNN, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)

KNN_new = grid.fit(X_train, Y_train) 

print(KNN_new.best_params_)

accuracy = KNN_new.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )

# Predict the class on test data
pred = KNN_new.predict(X_test)

pred

cm = skmet.confusion_matrix(Y_test, pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Benign', 'Malignant'])
cmplot.plot()
cmplot.ax_.set(title = 'Cancer Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

### Save the model
knn_best = KNN_new.best_estimator_
pickle.dump(knn_best, open('knn.pkl', 'wb'))

import os
os.getcwd()

# Load a saved model

model = pickle.load(open('knn.pkl', 'rb'))
   
