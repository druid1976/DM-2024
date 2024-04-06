import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.impute import SimpleImputer

titanic = pd.read_csv("train.csv")
categorical_features = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

#print(titanic.describe())
#print(titanic.info())


# need to save below processings to CSV file ???

# MinMax Scaling

numeric_cols = titanic.select_dtypes(include=[np.number]).columns
X = titanic[numeric_cols]

norm = MinMaxScaler(feature_range=(0,1)).fit(X)
X_minmax = pd.DataFrame(norm.transform(X), columns=X.columns)

# Standardization

scale = StandardScaler().fit(X)
X_scaled = pd.DataFrame(scale.transform(X), columns=X.columns)

print(X_minmax.describe().round(3))
print("\n\n")
print(X_scaled.describe().round(3))


# Imputing Data

imputer = SimpleImputer(strategy='mean')  # Replace with mean
X_imputed = imputer.fit_transform(X)

# Create normalizers for L-1 and L-2
normalizer_l1 = Normalizer(norm='l1')
normalizer_l2 = Normalizer(norm='l2')

# Normalize data using L-1 and L-2
normalized_data_l1 = normalizer_l1.fit_transform(X_imputed)
normalized_data_l2 = normalizer_l2.fit_transform(X_imputed)



# Observe the normalized data
print("\n\nOriginal data:\n", X[:5])  # Print the first 5 rows

print("\nL-1 normalized data:\n", normalized_data_l1[:5])
print("\nL-1 norm sum of each column:\n", normalized_data_l1.sum(axis=0))  # Check sum of absolute values

print("\n\nL-2 normalized data:\n", normalized_data_l2[:5])
print("\nL-2 norm sum of squares of each column:\n", (normalized_data_l2**2).sum(axis=0))  # Check sum of squares


"""
Why we don't use Discretization'
Limited Benefit for Most Features: In the Titanic dataset, many features (e.g., Age, Fare) likely already have a clear meaning and can be used directly in models. Discretization might not add much value.
Potential Information Loss: Discretization can group together data points that might have different behaviors, leading to information loss.
Choosing Cut Points: Defining meaningful cut points for discretization can be subjective and impact model performance.
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

titanic.to_csv("processed_titanic.csv", index=False)













