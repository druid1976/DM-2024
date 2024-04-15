from sklearn import datasets, feature_selection
from itertools import compress
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split


titanic = pd.read_csv("train.csv")
numeric_cols = titanic.select_dtypes(include=[np.number]).columns
numeric_titanic = titanic[numeric_cols]

print(numeric_titanic.std()**2)

print("\n\n\n")


#initialize it
scaler = MinMaxScaler()
#calculate all the necessary data to perform the normalization
scaler.fit(numeric_titanic)
#apply the standarizer to the data
titanic_normalized = pd.DataFrame(scaler.transform(numeric_titanic),columns = numeric_cols)
print(titanic_normalized.std()**2) # seems better


print("\n\n\n\n")
#we create a variance threshold object
sel = VarianceThreshold(threshold=0.03)
#we apply the method to the data - this method does not create a new dataframe, just a numpy array
titanic_variance_removed = sel.fit_transform(titanic_normalized)
#we can create a new dataframe using the get_feature_names_out() function to get the names of the selected columns
titanic_dataframe_variance_removed = pd.DataFrame(titanic_variance_removed, columns = sel.get_feature_names_out())
#print(titanic_dataframe_variance_removed)

print("\n\n\n\n")

print(sel.get_support())

print("\n\n\n\n")

print(titanic_normalized.corr())



#this just sets the size of a picture
plt.figure(figsize=(10,8))
#here we draw the heatmap
sns.heatmap(titanic_normalized.corr(), cmap='YlGnBu')


# Impute missing values using SimpleImputer (replace with your preferred strategy)
imputer = SimpleImputer(strategy='mean')
titanic_imputed_normalized = imputer.fit_transform(titanic_normalized)
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(titanic_imputed_normalized, y, test_size=0.2, random_state=42)

scores_chi2, p_vals_chi2 = chi2(X_train, y_train)
scores_anova, p_vals_anova = f_classif(X_train, y_train)
scores_mi = mutual_info_classif(X_train, y_train)
pd.DataFrame(scores_chi2, numeric_cols).plot(kind='barh', title='Chi squared')
pd.DataFrame(scores_anova, numeric_cols).plot(kind='barh', title='ANOVA')
pd.DataFrame(scores_mi, numeric_cols).plot(kind='barh', title='Mutual Information')


titanic_X = X_train[:, :-1]
# Feature selection using chi-squared for classification
skb_object = SelectKBest(chi2, k=2)
titanic_new = skb_object.fit(X_train, y_train)

# Transform X_test for consistency
titanic_new = skb_object.transform(X_test)

# Extract the selected features
feature1 = titanic_new[:, 0]
feature2 = titanic_new[:, 1]

# Plot the data, coloring by survival class
plt.plot(feature1[y_test  == 0], feature2[y_test  == 0], 'ro', alpha=0.5, label='Not Survived')
plt.plot(feature1[y_test  == 1], feature2[y_test  == 1], 'go', alpha=0.5, label='Survived')
"""
# Customize plot elements
plt.xlabel(skb_object.get_feature_names_out()[0])
plt.ylabel(skb_object.get_feature_names_out()[1])
plt.xlim([np.min(feature1) - 1, np.max(feature1) + 1])
plt.ylim([np.min(feature2) - 1, np.max(feature2) + 1])
plt.legend()
plt.show()
"""
















