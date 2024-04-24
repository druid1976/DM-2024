# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:47:20 2024

@author: EMRE KARATAŞ
"""
import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif


def feature_transform(titanic_data):
    """
    Below function converts categorical features "Embarked" and "Sex" from text labels into one-hot encoded numerical columns.
    """
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
    """
    Below function binarize the age column by assigning 1 if age grater than 18 and 0 otherwise
    """
    if age>=18:
        return 1 # Adult
    else:
        return 0 # Child
    
  
def binarize_lonliness(row):
    """
    Below function takes a row of data as input and checks the values in 'SibSp' (siblings/spouses onboard) and 
    'Parch' (parents/children onboard) columns. If the sum of these values is greater than 0, assigns 0 (Not Alone),
    otherwise assigns 1 (Alone). By this binarization IsAlone feature is formed which is more useful and easy to use
    """
    if row['SibSp']+row['Parch']>0:
        return 0 # Not Alone
    else:
        return 1 # Alone

    
   
def drop_features(titanic_data):
    """
    Below function is used for dropping unnecessary and redundant columns.
    These columns are analyzed in visualizations(correlation, feature selection graphs), binarized and encoded, that is
    why they are not useful anymore
    """ 
    return titanic_data.drop(['Embarked', 'Age','Parch','SibSp','Name','Ticket','Cabin','Sex','N'], axis=1,errors="ignore")



# don't forget to modify path of train.csv based on its location in your computer
titanic = pd.read_csv("train.csv")
numeric_cols = titanic.select_dtypes(include=[np.number]).columns
numeric_data = titanic[numeric_cols]
target = titanic['Survived']

print(titanic.info())
print("\n\n")
# Imputing Data
imputer = SimpleImputer(strategy='mean')  # Replace with mean
X_imputed = imputer.fit_transform(numeric_data)
titanic[numeric_cols] = X_imputed

#this just sets the size of a picture
plt.figure(figsize=(10,8))
#here we draw the heatmap, SibSp and Parch are corralated, so we can remove one of them. Or we can remove
# both since we have column as 'IsAlone'
sns.heatmap(numeric_data.corr(), cmap='YlGnBu')

# printing the graphs for feature selection
scores_chi2, p_vals_chi2 = chi2(X_imputed, target)
scores_anova, p_vals_anova = f_classif(X_imputed, target)
scores_mi = mutual_info_classif(X_imputed, target)
pd.DataFrame(scores_chi2, numeric_cols).plot(kind='barh', title='Chi squared')
pd.DataFrame(scores_anova, numeric_cols).plot(kind='barh', title='ANOVA')
pd.DataFrame(scores_mi, numeric_cols).plot(kind='barh', title='Mutual Information')

# printing before standardization
print(numeric_data.std()**2)
print('\n\n')

#initialize it
scaler = StandardScaler()
#calculate all the necessary data to perform the standarization
scaler.fit(X_imputed)
#apply the standarizer to the data
titanic_standardized = pd.DataFrame(scaler.transform(X_imputed),columns = numeric_cols)
titanic[numeric_cols] = titanic_standardized

# printing after standardization
print(titanic_standardized.std()**2)
print('\n\n')

# We can remove column 'Age', 'Parch', 'SibSp' since we binarized them
titanic['IsAdult'] = titanic['Age'].apply(binarize_age)
titanic['IsAlone'] = titanic.apply(binarize_lonliness, axis=1)

titanic = feature_transform(titanic)


"""
we encoded columns 'Embarked' and 'Sex' so we dont need them. Column 'N' is formed during OneHotEncoding represents none values so it is not relevant
because absenced values indicated by 0. 'Name','Ticket' and'Cabin' are categorical and not relevant to passanger survival
"""
titanic = drop_features(titanic)


X = titanic.drop(['Survived'],axis=1)
y = titanic['Survived']

# numerical features should be scaled, so that model can manage it easily
X_data = scaler.fit_transform(X)
# since scaler returns numpy array, we also need to y_data to be numpy array
y_data = y.to_numpy()

# data information after pre-processing, feature selection and feature extraction
print(titanic.info())

# uncomment below line if you want to save processed data for later use
#titanic.to_csv("processed_titanic.csv", index=False)
    



















