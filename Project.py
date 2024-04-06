import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

titanic = pd.read_csv("train.csv")

#print(titanic.describe())
#print(titanic.info())


numeric_cols = titanic.select_dtypes(include=[np.number]).columns
X = titanic[numeric_cols]

norm = MinMaxScaler(feature_range=(0,1)).fit(X)
X_minmax = pd.DataFrame(norm.transform(X), columns=X.columns)

scale = StandardScaler().fit(X)
X_scaled = pd.DataFrame(scale.transform(X), columns=X.columns)

"""
print(X_minmax.describe().round(3))
print("\n\n")
print(X_scaled.describe().round(3))
"""

"""
# Rows with NaN values are dropped below to be able to perform L1 and L2 Normalization
X_deleted = X.dropna()  # Drop rows with missing values

# Create normalizers for L-1 and L-2
normalizer_l1 = Normalizer(norm='l1')
normalizer_l2 = Normalizer(norm='l2')

# Normalize data using L-1 and L-2
data_l1 = normalizer_l1.fit_transform(X_deleted)
data_l2 = normalizer_l2.fit_transform(X_deleted)

# Observe the normalized data
print("Original data:\n", X[:5])  # Print the first 5 rows

print("\nL-1 normalized data:\n", data_l1[:5])
print("L-1 norm sum of each column:\n", data_l1.sum(axis=0))  # Check sum of absolute values

print("\nL-2 normalized data:\n", data_l2[:5])
print("L-2 norm sum of squares of each column:\n", (data_l2**2).sum(axis=0))  # Check sum of squares
"""

"""
Why we don't use Discretization'
Limited Benefit for Most Features: In the Titanic dataset, many features (e.g., Age, Fare) likely already have a clear meaning and can be used directly in models. Discretization might not add much value.
Potential Information Loss: Discretization can group together data points that might have different behaviors, leading to information loss.
Choosing Cut Points: Defining meaningful cut points for discretization can be subjective and impact model performance.
"""

"""
def binarize_age(age):
    if age>=18:
        return 1 # Adult
    else:
        return 0 # Child

titanic['IsAdult'] = titanic['Age'].apply(binarize_age)

def binarize_lonliness(row):
    if row['SibSp']+row['Parch']==0:
        return 1 # Alone
    elif row['SibSp']+row['Parch']>0:
        return 0 # Not Alone

titanic['IsAlone'] = titanic.apply(binarize_lonliness, axis=1)

print(titanic['IsAlone'])
"""

















