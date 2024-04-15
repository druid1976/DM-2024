# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:48:26 2024

@author: EMRE KARATAŞ
"""

import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv("train.csv")
numeric_cols = titanic.select_dtypes(include=[np.number]).columns
numeric_data = titanic[numeric_cols]

# Imputing Data

imputer = SimpleImputer(strategy='mean')  # Replace with mean
X_imputed = imputer.fit_transform(numeric_data)
titanic[numeric_cols] = X_imputed

#initialize it
scaler = StandardScaler()
#calculate all the necessary data to perform the standarization
scaler.fit(X_imputed)
#apply the standarizer to the data
titanic_standardized = pd.DataFrame(scaler.transform(X_imputed),columns = numeric_cols)
titanic[numeric_cols] = titanic_standardized

def feature_transform(titanic_data):
    encoder = OneHotEncoder()
    matrix = encoder.fit_transform(titanic_data[['Embarked']]).toarray()
    
    column_names = ['C','S','Q','N']
    
    for i in range(len(matrix.T)):
        titanic_data[column_names[i]] = matrix.T[i]
        
    matrix = encoder.fit_transform(titanic_data[['Sex']]).toarray()
    column_names = ['Female','Male']
    
    for i in range(len(matrix.T)):
        titanic_data[column_names[i]] = matrix.T[i]
        
    return titanic_data


def binarize_age(age):
    if age>=18:
        return 1 # Adult
    else:
        return 0 # Child
    

def binarize_lonliness(row):
    if row['SibSp']+row['Parch']==0:
        return 1 # Alone
    elif row['SibSp']+row['Parch']>0:
        return 0 # Not Alone
    
    
def drop_features(titanic_data):
    return titanic_data.drop(['Embarked','Name','Ticket','Cabin','Sex','N'], axis=1,errors="ignore")

titanic['IsAdult'] = titanic['Age'].apply(binarize_age)
titanic['IsAlone'] = titanic.apply(binarize_lonliness, axis=1)

titanic = feature_transform(titanic)
titanic = drop_features(titanic)
#cols = ['Parch','Fare']
#print(titanic[cols].tail())


print(numeric_data.std()**2)
print('\n\n')
print(titanic_standardized.std()**2)

#this just sets the size of a picture
plt.figure(figsize=(10,8))
#here we draw the heatmap
sns.heatmap(titanic_standardized.corr(), cmap='YlGnBu')


X = titanic.drop(['Survived'],axis=1)
y = titanic['Survived']

X_data = scaler.fit_transform(X)
y_data = y.to_numpy()

print('\n\n')

print(X_data)
print('\n\n')
print(y_data)




#print(titanic[titanic['PassengerId']==6].Age)
#titanic.to_csv("processed_titanic.csv", index=False)



















