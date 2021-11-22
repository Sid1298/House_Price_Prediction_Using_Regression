import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler

pd.set_option('display.expand_frame_repr', False)

training_data = pd.read_csv("train.csv")
# print(training_data.shape)

target = training_data['SalePrice']
# print(type(target), target.shape)

training_data.drop(['SalePrice'], axis=1, inplace=True)
# print(training_data.shape)


fig = plt.figure(figsize=(12, 12))
axes = fig.add_axes([0.1, 0.15, 0.8, 0.8])
sns.heatmap(training_data.isnull(), yticklabels=False, cbar=False)
plt.show()


features = [features for features in training_data.columns if training_data[features].isnull().sum() > 1]
missing_percentage = dict()

for feature in features:
    missing_percentage[feature] = np.round(training_data[feature].isnull().mean(), 4) * 100
    # print(feature, np.round(training_data[feature].isnull().mean(), 4)*100, "% missing values.\n")

# for item in missing_percentage.items():
#     print(item, end="\n")

# print([col for col in missing_percentage.items() if col[1] < 20])

# features_less_than_20 = [col[0] for col in missing_percentage.items() if col[1] > 20]

before = training_data.shape[1]
# print("Eliminating features with more than 20% missing data...")
training_data.drop([col[0] for col in missing_percentage.items() if col[1] > 20], axis=1, inplace=True)
after = training_data.shape[1]
# print(before-after, "Columns dropped.\n")
# Features with missing data eliminated.


fig = plt.figure(figsize=(12, 12))
axes = fig.add_axes([0.1, 0.15, 0.8, 0.8])
sns.heatmap(training_data.isnull(), yticklabels=False, cbar=False)
plt.show()


# Now that most missing data has been removed, exploring data further
integer_columns = training_data.select_dtypes(include=['int64']).columns
float_columns = training_data.select_dtypes(include=['float64']).columns
object_columns = training_data.select_dtypes(include=['object']).columns

# Categorical Features:
categorical_count = len(object_columns)
# print("found", categorical_count, "categorical columns.\n", sep=" ")
# print("Categorical columns are :", object_columns, "\n", sep=" ")
categorical_data = training_data[object_columns]
# categorical_data.describe().to_csv("categorical data info")
# columns with missing data
categorical_missing = categorical_data.columns[categorical_data.isnull().any()]
# print(len(categorical_missing), "columns have missing values", sep=" ")
# print("These are :", categorical_missing, sep=" ")

# print("\n", categorical_data.describe())
# Missing data can be imputed since its not a lot of missing values
categorical_imputer = SimpleImputer(missing_values=np.NaN, strategy="most_frequent")
for col in categorical_missing:
    categorical_data[col] = categorical_imputer.fit_transform(categorical_data[col].values.reshape(-1, 1))
    training_data[col] = categorical_imputer.fit_transform(training_data[col].values.reshape(-1, 1))

fig = plt.figure(figsize=(12, 12))
axes = fig.add_axes([0.1, 0.15, 0.8, 0.8])
sns.heatmap(training_data.isnull(), yticklabels=False, cbar=False)
plt.show()

# Numerical Features:

# print(training_data.columns[training_data.isnull().any()])
# ['LotFrontage', 'MasVnrArea', 'GarageYrBlt'] columns with missing values left to be imputed

plt.figure()
training_data['LotFrontage'].hist(bins=75)
plt.show()
# It makes sense to fill the missing data with random values from the observations. Better to chose such values near
# the mean to not affect distribution of the data
# print(training_data['LotFrontage'].describe())
# 4 <----- index of 25 %ile in pandas series
# 6 <----- index of 75 %ile in pandas series

# print(training_data['LotFrontage'].describe()[4], training_data['LotFrontage'].describe()[6])
training_data['LotFrontage'].fillna(np.random.randint(training_data['LotFrontage'].describe()[4],
                                                      training_data['LotFrontage'].describe()[6]),
                                    inplace=True)

fig = plt.figure(figsize=(12, 12))
axes1 = fig.add_axes([0.1, 0.15, 0.8, 0.8])
sns.heatmap(training_data.isnull(), yticklabels=False, cbar=False)
plt.show()
# Values are filled.

plt.figure()
training_data['MasVnrArea'].hist(bins=75)
plt.show()
# Most values are 0 in this feature, it makes more sense to fill the values with 0
training_data['MasVnrArea'].fillna(0, inplace=True)

fig = plt.figure(figsize=(12, 12))
axes2 = fig.add_axes([0.1, 0.15, 0.8, 0.8])
sns.heatmap(training_data.isnull(), yticklabels=False, cbar=False)
plt.show()
# Values are filled

# print(training_data['GarageYrBlt'].isna().sum())

# print(training_data['GarageYrBlt'].describe())

plt.figure()
training_data['GarageYrBlt'].hist(bins=75)
plt.show()
# This is date related data, while it is hard to impute, it is visible that most data lies in the 25 %ile to maximum
# value range. So it makes sense to either drop this feature or choose random values from the 25 %ile to max value range
# I am choosing to impute the data using random values
training_data['GarageYrBlt'].fillna(np.random.randint(training_data['GarageYrBlt'].describe()[4],
                                                      training_data['GarageYrBlt'].describe()[7]),
                                    inplace=True)

fig = plt.figure(figsize=(12, 12))
axes3 = fig.add_axes([0.1, 0.15, 0.8, 0.8])
sns.heatmap(training_data.isnull(), yticklabels=False, cbar=False)
plt.show()
# Values are filled

# All data in training set has been imputed

# Lets now clean the test data
testing_data = pd.read_csv("test.csv")
testing_data.drop([col[0] for col in missing_percentage.items() if col[1] > 20], axis=1, inplace=True)

# print(testing_data.info())

for col in categorical_missing:
    testing_data[col] = testing_data[col].fillna(training_data[col].mode()[0])

testing_data['LotFrontage'].fillna(np.random.randint(training_data['LotFrontage'].describe()[4],
                                                     training_data['LotFrontage'].describe()[6]),
                                   inplace=True)
testing_data['MasVnrArea'].fillna(0, inplace=True)
testing_data['GarageYrBlt'].fillna(np.random.randint(training_data['GarageYrBlt'].describe()[4],
                                                     training_data['GarageYrBlt'].describe()[7]),
                                   inplace=True)

# testing data cleaned. filled all features with imputed values from training data.

# Now lets explore relations of various features with SalePrice, ie. target variable
# print(target.describe())

plt.figure()
sns.histplot(data=target, kde=True, bins=75)
# plt.show()

# The distance from the street of the actual property can be factor of the pricing of the house. Generally there is a
# sweet spot of sorts, where large distances can make the house feel "disconnected" while short distances can make the
# house feel too close to the street.
data = pd.concat([target, training_data['LotFrontage']], axis=1)
data.plot.scatter(x='LotFrontage', y='SalePrice', ylim=(0, 900000))
# plt.show()

# Usually, living area has an impact on pricing of houses. This includes the square footage of the living area
# and basement area
data = pd.concat([target, training_data['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 900000))
# plt.show()

data = pd.concat([target, training_data['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 900000))
# plt.show()

# The overall quality of the house is a useful feature of this dataset as it is a quantification of the quality of the
# house. A fairly linear relation is expected.
data = pd.concat([target, training_data['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
# plt.show()

# Checking categorical features for encoding
fig1, axes = plt.subplots(round(len(object_columns)/3), 3, figsize=(24, 60))
for i, ax in enumerate(fig1.axes):
    if i < len(object_columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=object_columns[i], alpha=0.7, data=training_data.select_dtypes(include=['object']), ax=ax)
fig1.tight_layout()
plt.show()

# Some features in the data have certain skews in a way that creating dummy variables / encoding them will create more,
# unwanted features and increase dimensionality. So approach can be changed, to encode these variables rather than
# creating dummies for all categories.
object_columns = training_data.select_dtypes(include=['object']).columns
value_counts = dict()
for col in object_columns:
    value_counts[col] = len(training_data[col].unique())

# c = 0
# for label, count in value_counts.items():
#     if count == 2:
#         c += 1
#     print(label, ":", count)

# print(c, len(value_counts))

# print(categorical_data['Condition1'].value_counts().sort_values(ascending=False))
# Some values in the feature have a very few occurrences and encoding them will add unnecessary dimensions to the data.
# So it makes sense to encode the more frequent labels in the data. Encoding 4 most frequent labels will help in
# encoding most data while keeping fewer dimensions in the data.
frequent_labels = [x for x in categorical_data['Condition1'].value_counts().sort_values(ascending=False).head(4).index]
# print(frequent_labels)
for label in frequent_labels:
    categorical_data[label] = np.where(categorical_data['Condition1'] == label, 1, 0)
# print(categorical_data[['Condition1']+frequent_labels].head(10))
# print(data)


def one_hot_frequency_labels(data, variable, freq_labels):
    """
    :param data: dataframe to be encoded
    :param variable: feature in dataframe to be encoded
    :param freq_labels: value in the feature being encoded to 1/0 for presence/absence
    :return: void, inplace changes
    """
    for label in freq_labels:
        data[variable+"_"+label] = np.where(data[variable] == label, 1, 0)

"""
MSZoning 		:3
Street 		:1
LotShape 		:3
LandContour 		:2
Utilities 		:1
LotConfig 		:3
LandSlope 		:2
Condition1 		:4
Condition2 		:2
BldgType 		:3
HouseStyle 		:4
RoofStyle 		:2
RoofMatl 		:2
Exterior1st 		:5
Exterior2nd 		:5
MasVnrType 		:3
ExterQual 		:3
ExterCond 		:3
Foundation 		:3
BsmtQual 		:4
BsmtCond 		:3
BsmtExposure 		:3
BsmtFinType1 		:6
BsmtFinType2 		:2
Heating 		:1
HeatingQC 		:3
CentralAir 		:1
Electrical 		:2
KitchenQual 		:3
Functional 		:2
GarageType 		:3
GarageFinish 		:3
GarageQual 		:2
GarageCond 		:2
PavedDrive 		:2
SaleType 		:3
SaleCondition 		:3
Neighborhood 		: For this feature, I will do further examination of the labels.
"""

data = pd.concat([target, training_data['Neighborhood']], axis=1)
f, ax = plt.subplots(figsize=(25, 15))
fig = sns.boxplot(x='Neighborhood', y='SalePrice', data=data)
plt.show()

# There seems to be a lot of variation among the 25 values in this feature, and all of them seem to have different
# affect on the target feature. It makes more sense to encode these values with their respective probabilities but I am
# choosing to create dummy variables for the ten most frequent labels for the sake of simplicity.
# Lets start start with Neighborhood feature and then encode other features as per their chosen most frequent values.

frequent_labels_Neighborhood = [x for x in categorical_data['Neighborhood'].value_counts().sort_values(ascending=False
                                                                                                       ).head(10).index]
one_hot_frequency_labels(training_data, 'Neighborhood', frequent_labels_Neighborhood)
one_hot_frequency_labels(testing_data, 'Neighborhood', frequent_labels_Neighborhood)

# selecting number of features for encoding and keeping dimensionality in check.
# These labels have been chosen manually.

frequent_labels_MSZoning = [x for x in categorical_data['MSZoning'].value_counts().sort_values(ascending=False
                                                                                               ).head(3).index]
one_hot_frequency_labels(training_data, 'MSZoning', frequent_labels_MSZoning)
one_hot_frequency_labels(testing_data, 'MSZoning', frequent_labels_MSZoning)

frequent_labels_Street = [x for x in categorical_data['Street'].value_counts().sort_values(ascending=False
                                                                                           ).head(1).index]
one_hot_frequency_labels(training_data, 'Street', frequent_labels_Street)
one_hot_frequency_labels(testing_data, 'Street', frequent_labels_Street)

frequent_labels_LotShape = [x for x in categorical_data['LotShape'].value_counts().sort_values(ascending=False
                                                                                               ).head(3).index]
one_hot_frequency_labels(training_data, 'LotShape', frequent_labels_LotShape)
one_hot_frequency_labels(testing_data, 'LotShape', frequent_labels_LotShape)

frequent_labels_LandContour = [x for x in categorical_data['LandContour'].value_counts().sort_values(ascending=False
                                                                                                     ).head(2).index]
one_hot_frequency_labels(training_data, 'LandContour', frequent_labels_LandContour)
one_hot_frequency_labels(testing_data, 'LandContour', frequent_labels_LandContour)

frequent_labels_Utilities = [x for x in categorical_data['Utilities'].value_counts().sort_values(ascending=False
                                                                                                 ).head(1).index]
one_hot_frequency_labels(training_data, 'Utilities', frequent_labels_Utilities)
one_hot_frequency_labels(testing_data, 'Utilities', frequent_labels_Utilities)

frequent_labels_LotConfig = [x for x in categorical_data['LotConfig'].value_counts().sort_values(ascending=False
                                                                                                 ).head(3).index]
one_hot_frequency_labels(training_data, 'LotConfig', frequent_labels_LotConfig)
one_hot_frequency_labels(testing_data, 'LotConfig', frequent_labels_LotConfig)

frequent_labels_LandSlope = [x for x in categorical_data['LandSlope'].value_counts().sort_values(ascending=False
                                                                                                 ).head(2).index]
one_hot_frequency_labels(training_data, 'LandSlope', frequent_labels_LandSlope)
one_hot_frequency_labels(testing_data, 'LandSlope', frequent_labels_LandSlope)

frequent_labels_Condition1 = [x for x in categorical_data['Condition1'].value_counts().sort_values(ascending=False
                                                                                                   ).head(4).index]
one_hot_frequency_labels(training_data, 'Condition1', frequent_labels_Condition1)
one_hot_frequency_labels(testing_data, 'Condition1', frequent_labels_Condition1)

frequent_labels_Condition2 = [x for x in categorical_data['Condition2'].value_counts().sort_values(ascending=False
                                                                                                   ).head(2).index]
one_hot_frequency_labels(training_data, 'Condition2', frequent_labels_Condition2)
one_hot_frequency_labels(testing_data, 'Condition2', frequent_labels_Condition2)

frequent_labels_BldgType = [x for x in categorical_data['BldgType'].value_counts().sort_values(ascending=False
                                                                                               ).head(3).index]
one_hot_frequency_labels(training_data, 'BldgType', frequent_labels_BldgType)
one_hot_frequency_labels(testing_data, 'BldgType', frequent_labels_BldgType)

frequent_labels_HouseStyle = [x for x in categorical_data['HouseStyle'].value_counts().sort_values(ascending=False
                                                                                                   ).head(4).index]
one_hot_frequency_labels(training_data, 'HouseStyle', frequent_labels_HouseStyle)
one_hot_frequency_labels(testing_data, 'HouseStyle', frequent_labels_HouseStyle)

frequent_labels_RoofStyle = [x for x in categorical_data['RoofStyle'].value_counts().sort_values(ascending=False
                                                                                                 ).head(2).index]
one_hot_frequency_labels(training_data, 'RoofStyle', frequent_labels_RoofStyle)
one_hot_frequency_labels(testing_data, 'RoofStyle', frequent_labels_RoofStyle)

frequent_labels_RoofMatl = [x for x in categorical_data['RoofMatl'].value_counts().sort_values(ascending=False
                                                                                               ).head(2).index]
one_hot_frequency_labels(training_data, 'RoofMatl', frequent_labels_RoofMatl)
one_hot_frequency_labels(testing_data, 'RoofMatl', frequent_labels_RoofMatl)

frequent_labels_Exterior1st = [x for x in categorical_data['Exterior1st'].value_counts().sort_values(ascending=False
                                                                                                     ).head(5).index]
one_hot_frequency_labels(training_data, 'Exterior1st', frequent_labels_Exterior1st)
one_hot_frequency_labels(testing_data, 'Exterior1st', frequent_labels_Exterior1st)

frequent_labels_Exterior2nd = [x for x in categorical_data['Exterior2nd'].value_counts().sort_values(ascending=False
                                                                                                     ).head(5).index]
one_hot_frequency_labels(training_data, 'Exterior2nd', frequent_labels_Exterior2nd)
one_hot_frequency_labels(testing_data, 'Exterior2nd', frequent_labels_Exterior2nd)

frequent_labels_MasVnrType = [x for x in categorical_data['MasVnrType'].value_counts().sort_values(ascending=False
                                                                                                   ).head(3).index]
one_hot_frequency_labels(training_data, 'MasVnrType', frequent_labels_MasVnrType)
one_hot_frequency_labels(testing_data, 'MasVnrType', frequent_labels_MasVnrType)

frequent_labels_ExterQual = [x for x in categorical_data['ExterQual'].value_counts().sort_values(ascending=False
                                                                                                 ).head(3).index]
one_hot_frequency_labels(training_data, 'ExterQual', frequent_labels_ExterQual)
one_hot_frequency_labels(testing_data, 'ExterQual', frequent_labels_ExterQual)

frequent_labels_ExterCond = [x for x in categorical_data['ExterCond'].value_counts().sort_values(ascending=False
                                                                                                 ).head(3).index]
one_hot_frequency_labels(training_data, 'ExterCond', frequent_labels_ExterCond)
one_hot_frequency_labels(testing_data, 'ExterCond', frequent_labels_ExterCond)

frequent_labels_Foundation = [x for x in categorical_data['Foundation'].value_counts().sort_values(ascending=False
                                                                                                   ).head(3).index]
one_hot_frequency_labels(training_data, 'Foundation', frequent_labels_Foundation)
one_hot_frequency_labels(testing_data, 'Foundation', frequent_labels_Foundation)

frequent_labels_BsmtQual = [x for x in categorical_data['BsmtQual'].value_counts().sort_values(ascending=False
                                                                                               ).head(4).index]
one_hot_frequency_labels(training_data, 'BsmtQual', frequent_labels_BsmtQual)
one_hot_frequency_labels(testing_data, 'BsmtQual', frequent_labels_BsmtQual)

frequent_labels_BsmtCond = [x for x in categorical_data['BsmtCond'].value_counts().sort_values(ascending=False
                                                                                               ).head(3).index]
one_hot_frequency_labels(training_data, 'BsmtCond', frequent_labels_BsmtCond)
one_hot_frequency_labels(testing_data, 'BsmtCond', frequent_labels_BsmtCond)

frequent_labels_BsmtExposure = [x for x in categorical_data['BsmtExposure'].value_counts().sort_values(ascending=False
                                                                                                       ).head(3).index]
one_hot_frequency_labels(training_data, 'BsmtExposure', frequent_labels_BsmtExposure)
one_hot_frequency_labels(testing_data, 'BsmtExposure', frequent_labels_BsmtExposure)

frequent_labels_BsmtFinType1 = [x for x in categorical_data['BsmtFinType1'].value_counts().sort_values(ascending=False
                                                                                                       ).head(6).index]
one_hot_frequency_labels(training_data, 'BsmtFinType1', frequent_labels_BsmtFinType1)
one_hot_frequency_labels(testing_data, 'BsmtFinType1', frequent_labels_BsmtFinType1)

frequent_labels_BsmtFinType2 = [x for x in categorical_data['BsmtFinType2'].value_counts().sort_values(ascending=False
                                                                                                       ).head(2).index]
one_hot_frequency_labels(training_data, 'BsmtFinType2', frequent_labels_BsmtFinType2)
one_hot_frequency_labels(testing_data, 'BsmtFinType2', frequent_labels_BsmtFinType2)

frequent_labels_Heating = [x for x in categorical_data['Heating'].value_counts().sort_values(ascending=False
                                                                                             ).head(1).index]
one_hot_frequency_labels(training_data, 'Heating', frequent_labels_Heating)
one_hot_frequency_labels(testing_data, 'Heating', frequent_labels_Heating)

frequent_labels_HeatingQC = [x for x in categorical_data['HeatingQC'].value_counts().sort_values(ascending=False
                                                                                                 ).head(3).index]
one_hot_frequency_labels(training_data, 'HeatingQC', frequent_labels_HeatingQC)
one_hot_frequency_labels(testing_data, 'HeatingQC', frequent_labels_HeatingQC)

frequent_labels_CentralAir = [x for x in categorical_data['CentralAir'].value_counts().sort_values(ascending=False
                                                                                                   ).head(1).index]
one_hot_frequency_labels(training_data, 'CentralAir', frequent_labels_CentralAir)
one_hot_frequency_labels(testing_data, 'CentralAir', frequent_labels_CentralAir)

frequent_labels_Electrical = [x for x in categorical_data['Electrical'].value_counts().sort_values(ascending=False
                                                                                                   ).head(2).index]
one_hot_frequency_labels(training_data, 'Electrical', frequent_labels_Electrical)
one_hot_frequency_labels(testing_data, 'Electrical', frequent_labels_Electrical)

frequent_labels_KitchenQual = [x for x in categorical_data['KitchenQual'].value_counts().sort_values(ascending=False
                                                                                                     ).head(3).index]
one_hot_frequency_labels(training_data, 'KitchenQual', frequent_labels_KitchenQual)
one_hot_frequency_labels(testing_data, 'KitchenQual', frequent_labels_KitchenQual)

frequent_labels_Functional = [x for x in categorical_data['Functional'].value_counts().sort_values(ascending=False
                                                                                                   ).head(2).index]
one_hot_frequency_labels(training_data, 'Functional', frequent_labels_Functional)
one_hot_frequency_labels(testing_data, 'Functional', frequent_labels_Functional)

frequent_labels_GarageType = [x for x in categorical_data['GarageType'].value_counts().sort_values(ascending=False
                                                                                                   ).head(3).index]
one_hot_frequency_labels(training_data, 'GarageType', frequent_labels_GarageType)
one_hot_frequency_labels(testing_data, 'GarageType', frequent_labels_GarageType)

frequent_labels_GarageFinish = [x for x in categorical_data['GarageFinish'].value_counts().sort_values(ascending=False
                                                                                                       ).head(3).index]
one_hot_frequency_labels(training_data, 'GarageFinish', frequent_labels_GarageFinish)
one_hot_frequency_labels(testing_data, 'GarageFinish', frequent_labels_GarageFinish)

frequent_labels_GarageQual = [x for x in categorical_data['GarageQual'].value_counts().sort_values(ascending=False
                                                                                                   ).head(2).index]
one_hot_frequency_labels(training_data, 'GarageQual', frequent_labels_GarageQual)
one_hot_frequency_labels(testing_data, 'GarageQual', frequent_labels_GarageQual)

frequent_labels_GarageCond = [x for x in categorical_data['GarageCond'].value_counts().sort_values(ascending=False
                                                                                                   ).head(2).index]
one_hot_frequency_labels(training_data, 'GarageCond', frequent_labels_GarageCond)
one_hot_frequency_labels(testing_data, 'GarageCond', frequent_labels_GarageCond)

frequent_labels_PavedDrive = [x for x in categorical_data['PavedDrive'].value_counts().sort_values(ascending=False
                                                                                                   ).head(2).index]
one_hot_frequency_labels(training_data, 'PavedDrive', frequent_labels_PavedDrive)
one_hot_frequency_labels(testing_data, 'PavedDrive', frequent_labels_PavedDrive)

frequent_labels_SaleType = [x for x in categorical_data['SaleType'].value_counts().sort_values(ascending=False
                                                                                               ).head(3).index]
one_hot_frequency_labels(training_data, 'SaleType', frequent_labels_SaleType)
one_hot_frequency_labels(testing_data, 'SaleType', frequent_labels_SaleType)

frequent_labels_SaleCondition = [x for x in categorical_data['SaleCondition'].value_counts().sort_values(ascending=False
                                                                                                         ).head(3
                                                                                                                ).index]
one_hot_frequency_labels(training_data, 'SaleCondition', frequent_labels_SaleCondition)
one_hot_frequency_labels(testing_data, 'SaleCondition', frequent_labels_SaleCondition)

# now drop original categorical columns from both datasets and check for missing values if any
training_data.drop(object_columns, axis=1, inplace=True)
testing_data.drop(object_columns, axis=1, inplace=True)

fig1 = plt.figure(figsize=(12, 12))
axes = fig1.add_axes([0.1, 0.15, 0.8, 0.8])
sns.heatmap(training_data.isnull(), yticklabels=False, cbar=False)
plt.show()

testing_data.dropna(inplace=True)
# deleted three rows with remaining missing values. This much data can bre dropped as the loss of any information is
# insignificant

fig1 = plt.figure(figsize=(12, 12))
axes = fig1.add_axes([0.1, 0.15, 0.8, 0.8])
sns.heatmap(testing_data.isnull(), yticklabels=False, cbar=False)
plt.show()

# Time for the fun part. Importing libraries and performing regression to make predictions.

report = open("report.txt", "w+")


def model_scores(true, model):
    predicted = model.predict(training_data)
    r2_squared = metrics.r2_score(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    mae = metrics.mean_absolute_error(true, predicted)
    report.writelines(f"""\nTraining set score: {model.score(training_data, target)}
    _____________________________________\n
    Mean Absolute Error: {mae}\n
    Mean Squared Error: {mse}\n
    Root Mean Squared Error: {np.sqrt(mse)}\n
    R2 squared:{r2_squared}\n
    _____________________________________\n\n\n""")


linear_model = LinearRegression()
bagging_regressor = BaggingRegressor()
decision_tree_regressor = DecisionTreeRegressor()
ada_boost_regressor = AdaBoostRegressor(random_state=5)
random_forest_regressor = RandomForestRegressor(random_state=42)
gradient_boosting_regressor = GradientBoostingRegressor(random_state=123)

# Linear Regression
# _____________________________________________________________________________________________________________________
linear_model.fit(training_data, target)
test_predict_lr = linear_model.predict(testing_data)
report.writelines("------------------------Linear Regression------------------------")
model_scores(target, linear_model)

# Bagging Regression
# _____________________________________________________________________________________________________________________
bagging_regressor.fit(training_data, target)
test_predict_bg = bagging_regressor.predict(testing_data)
report.writelines("------------------------Bagging Regression------------------------")
model_scores(target, bagging_regressor)

# Decision Tree Regression
# _____________________________________________________________________________________________________________________
decision_tree_regressor.fit(training_data, target)
test_predict_dt = decision_tree_regressor.predict(testing_data)
report.writelines("------------------------Decision Tree Regression------------------------")
model_scores(target, decision_tree_regressor)

# ADA Boost Regression
# _____________________________________________________________________________________________________________________
ada_boost_regressor.fit(training_data, target)
test_predict_ada = ada_boost_regressor.predict(testing_data)
report.writelines("------------------------ADA Boost Regression------------------------")
model_scores(target, ada_boost_regressor)

# Random Forest Regression
# _____________________________________________________________________________________________________________________
random_forest_regressor.fit(training_data, target)
test_predict_rf = random_forest_regressor.predict(testing_data)
report.writelines("------------------------Random Forest Regression------------------------")
model_scores(target, random_forest_regressor)

# Gradient Boosting Regression
# _____________________________________________________________________________________________________________________
gradient_boosting_regressor.fit(training_data, target)
test_predict_gb = gradient_boosting_regressor.predict(testing_data)
report.writelines("------------------------Gradient Boosting Regression------------------------")
model_scores(target, gradient_boosting_regressor)

report.close()
