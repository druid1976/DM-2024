from sklearn import datasets, feature_selection
from itertools import compress
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

titanic = pd.read_csv("train.csv")
numeric_cols = titanic.select_dtypes(include=[np.number]).columns
numeric_titanic = titanic[numeric_cols]

print(numeric_titanic.std()**2)

print("\n\n\n")

#initialize it
scaler = StandardScaler()
#calculate all the necessary data to perform the standarization
scaler.fit(numeric_titanic)
#apply the standarizer to the data
titanic_standardized = pd.DataFrame(scaler.transform(numeric_titanic),columns = numeric_cols)
print(titanic_standardized.std()**2)

print("\n\n\n")
print(titanic_standardized.head())

print("\n\n\n")
#initialize it
scaler = MinMaxScaler()
#calculate all the necessary data to perform the normalization
scaler.fit(numeric_titanic)
#apply the standarizer to the data
titanic_normalized = pd.DataFrame(scaler.transform(numeric_titanic),columns = numeric_cols)
print(titanic_normalized.std()**2) # seems better

print("\n\n\n")
print(titanic_normalized.head())

print("\n\n\n\n")
#we create a variance threshold object
sel = VarianceThreshold(threshold=0.03)
#we apply the method to the data - this method does not create a new dataframe, just a numpy array
titanic_variance_removed = sel.fit_transform(titanic_normalized)
#we can create a new dataframe using the get_feature_names_out() function to get the names of the selected columns
titanic_dataframe_variance_removed = pd.DataFrame(titanic_variance_removed, columns = sel.get_feature_names_out())
print(titanic_dataframe_variance_removed)

print("\n\n\n\n")

print(titanic_normalized.corr())


