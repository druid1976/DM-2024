# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:48:26 2024

@author: EMRE KARATAŞ
"""

import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

titanic = pd.read_csv("train.csv")

numeric_cols = titanic.select_dtypes(include=[np.number]).columns
X = titanic[numeric_cols]

# Imputing Data

imputer = SimpleImputer(strategy='mean')  # Replace with mean
X_imputed = imputer.fit_transform(X)
titanic[numeric_cols] = X_imputed

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

titanic['IsAdult'] = titanic['Age'].apply(binarize_age)
titanic['IsAlone'] = titanic.apply(binarize_lonliness, axis=1)

titanic = feature_transform(titanic)


#print(titanic[titanic['PassengerId']==6].Age)
#titanic.to_csv("processed_titanic.csv", index=False)


"""
# MinMax Scaling
norm = MinMaxScaler(feature_range=(0,1)).fit(X_imputed)
X_minmax = pd.DataFrame(norm.transform(X_imputed), columns=X.columns)

# Standardization
scale = StandardScaler().fit(X_imputed)
X_scaled = pd.DataFrame(scale.transform(X_imputed), columns=X.columns)

print(X_minmax.describe().round(3))
print("\n\n")
print(X_scaled.describe().round(3))
"""

















