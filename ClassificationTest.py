# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:23:17 2024

@author: EMRE
"""



from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

classifier = SVC()

pd.options.mode.copy_on_write = True
titanic = pd.read_csv('train.csv')


# please run the first commented part to see results of unprocessed data


X = titanic[['Pclass','Age','Fare','Sex']]
X['Sex'] = X.Sex.copy().apply(lambda x : 1.0 if x == 'male' else 2.0).copy()

X = X.fillna(0)
X = X.to_numpy()

y = titanic['Survived']
y = y.to_numpy()

test_data = pd.read_csv('test.csv')

X_test = titanic[['Pclass','Age','Fare','Sex']]
X_test.Sex = X_test.Sex.apply(lambda x : 1.0 if x == 'male' else 2.0)
X_test = X_test.fillna(0)
X_test = X_test.to_numpy()
y_test = titanic['Survived']
y_test = y_test.to_numpy()

classifier.fit(X,y)
predicts = classifier.predict(X_test)

print(classification_report(y_test, predicts))


# then run the below commented part to see results of processed data

"""
numeric_cols = titanic.select_dtypes(include=[np.number]).columns
numeric_data = titanic[numeric_cols]
target = titanic['Survived']

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
    if row['SibSp']+row['Parch']>0:
        return 0 # Not Alone
    else:
        return 1 # Alone

    
    
def drop_features(titanic_data):
    return titanic_data.drop(['Embarked', 'Age','Parch','SibSp','Name','Ticket','Cabin','Sex','N'], axis=1,errors="ignore")

# We can remove column 'Age' since we binarized it
titanic['IsAdult'] = titanic['Age'].apply(binarize_age)
titanic['IsAlone'] = titanic.apply(binarize_lonliness, axis=1)


#this just sets the size of a picture
plt.figure(figsize=(10,8))
#here we draw the heatmap, SibSp and Parch are corralated, so we can remove one of them. Or we can remove
# both since we have column as 'IsAlone'
sns.heatmap(titanic_standardized.corr(), cmap='YlGnBu')


titanic = feature_transform(titanic)


scores_chi2, p_vals_chi2 = chi2(X_imputed, target)
scores_anova, p_vals_anova = f_classif(X_imputed, target)
scores_mi = mutual_info_classif(X_imputed, target)
pd.DataFrame(scores_chi2, numeric_cols).plot(kind='barh', title='Chi squared')
pd.DataFrame(scores_anova, numeric_cols).plot(kind='barh', title='ANOVA')
pd.DataFrame(scores_mi, numeric_cols).plot(kind='barh', title='Mutual Information')


# we encoded columns 'Embarked' and 'Sex' so we dont need them. Column 'N' is formed during OneHotEncoding represents none values so it is not relevant
# because absenced values indicated by 0, columns 'Parch','SibSp' are binarized. 'Name','Ticket' and'Cabin' are categorical and not relevant to passanger survival
titanic = drop_features(titanic)


X = titanic.drop(['Survived'],axis=1)
y = titanic['Survived']

X_data = scaler.fit_transform(X)
# Encode the target variable
le = LabelEncoder()

X = X.to_numpy()

#The SVM classifier (SVC) expects categorical labels (0 or 1) for the target variable (Survived), 
#but it receives continuous values when target data is numpy array. That is why Label encoding is implemented

y_data = le.fit_transform(y)
#y_data = y.to_numpy()

test_data = pd.read_csv('test.csv')

X_test = titanic.drop(['Survived'],axis=1)
X_test = X_test.fillna(0)
X_test = X_test.to_numpy()
y_test = titanic['Survived']
y_test = le.transform(y_test)
#y_test = y_test.to_numpy()

classifier.fit(X,y_data)
predicts = classifier.predict(X_test)

print(classification_report(y_test, predicts))
"""


















