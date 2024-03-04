import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ucimlrepo import fetch_ucirepo

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import xgboost as xgb

##################
## Load the Dataset

adult = fetch_ucirepo(id=2) 

X = adult.data.features 
y = adult.data.targets 

# Create dataframe from uci
df = X.join(y)


##################
## Show data info

print('\n================== dataframe ==================')
display(df)

print('\n================== metadata ==================')
print(adult.metadata) 
  
print('\n================== variables ==================')
print(adult.variables)

print('\n================== info ==================')
print(df.info())



##################
## Data Cleaning

print('\n================== shape before cleaning ==================')
print(df.shape)

# Change '?' to NA so those rows can be dropped in the next step
df = df.replace('?', pd.NA)

# Drop rows containing NA
df = df.dropna()

print('\n================== shape after dropping NA ==================')
print(df.shape)

# 'education' is a redundant column
# 'education-num' is the ordinally encoded column representing 'education'
df = df.drop('education', axis=1)

# 'income' has different formats for its data
# fix this column to make it consistent
df['income'] = df['income'].replace('>50K.', '>50K')
df['income'] = df['income'].replace('<=50K.', '<=50K')

print('\n================== final shape after cleaning ==================')
print(df.shape)


##################
## Data Preprocessing

# Binarize Target Variable
df['income'] = np.where(df['income'] == '>50K', 1, 0)
df.rename(columns={'income':'income>50K'}, inplace=True)

# Show reformatted target variable
print('\n================== target variable column ==================')
display(df['income>50K'])

# Split features from target
y3 = df.iloc[:,-1:]
X3 = df.iloc[:,1:-1]

# Aggregration - marital-status
X3['marital-status'].replace(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married', inplace=True)
X3['marital-status'].replace(['Divorced', 'Separated', 'Widowed'], 'Divorced', inplace=True)

# Aggregration - workclass
X3['workclass'].replace(['Federal-gov', 'Local-gov', 'State-gov'], 'government', inplace=True)
X3['workclass'].replace(['Never-worked', 'Without-pay'], 'jobless', inplace=True)

# Aggregration - occupation
X3['occupation'].replace(['Tech-support', 'Craft-repair', 'Machine-op-inspct'], 'Technical/Support', inplace=True)
X3['occupation'].replace(['Other-service', 'Priv-house-serv', 'Protective-serv'], 'Service', inplace=True)
X3['occupation'].replace(['Exec-managerial', 'Adm-clerical'], 'Management/Administration', inplace=True)
X3['occupation'].replace(['Handlers-cleaners', 'Farming-fishing', 'Transport-moving'], 'Manual Labor', inplace=True)

# Aggregration - relationship
X3['relationship'].replace(['Wife', 'Husband'], 'Spouse', inplace=True)
X3['relationship'].replace(['Not-in-family', 'Unmarried'], 'Non-Family', inplace=True)

# Aggregration - native-country
X3['native-country'].replace(['United-States', 'Canada', 'Outlying-US(Guam-USVI-etc)'], 'North America', inplace=True)
X3['native-country'].replace(['England', 'Germany', 'Greece', 'Italy', 'Poland', 'Portugal', 'Ireland', 'France', 'Scotland', 'Yugoslavia', 'Hungary', 'Holand-Netherlands'], 'Europe', inplace=True)
X3['native-country'].replace(['Cambodia', 'India', 'Japan', 'China', 'Philippines', 'Vietnam', 'Taiwan', 'Laos', 'Iran', 'Thailand', 'Hong'], 'Asia', inplace=True)
X3['native-country'].replace(['Ecuador', 'Columbia', 'Peru','Puerto-Rico', 'Mexico', 'Cuba', 'Jamaica', 'Dominican-Republic', 'Haiti', 'Guatemala', 'Honduras', 'El-Salvador', 'Nicaragua', 'Trinadad&Tobago', 'Panama'], 'Central & South America', inplace=True)

# One-Hot-Encoding for categorical columns
X3 = pd.get_dummies(X3, columns=['marital-status'], dtype=int)
X3 = pd.get_dummies(X3, columns=['workclass'], dtype=int)
X3 = pd.get_dummies(X3, columns=['occupation'], dtype=int)
X3 = pd.get_dummies(X3, columns=['relationship'], dtype=int)
X3 = pd.get_dummies(X3, columns=['race'], dtype=int)
X3 = pd.get_dummies(X3, columns=['native-country'], dtype=int)

# Encode sex 1 or 0
X3['sex'] = np.where(X3['sex'] == 'Male', 1, 0)

print('\n================== features dataframe after preprocessing ==================')
display(X3.head())

print('\n================== features dataframe shape after preprocessing ==================')
print(X3.shape)


##################
## Classifier Implementation

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=1)

##################
## Logistic Regression

print('\n================== Logistic Regression ==================')
print('\n======================================================')

# Create a pipeline that standardizes the data then creates a Logistic Regression model
pipeline = Pipeline([
    ('scaler', StandardScaler()),               # Feature scaling
    ('logreg', LogisticRegression(max_iter=1000))  # Logistic Regression classifier
])

# Define the parameter grid
param_grid = {
    'logreg__C': np.logspace(-4, 4, 20),               # Regularization strength
    'logreg__solver': ['liblinear', 'lbfgs'],  # Solvers
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print('\n================== Best parameters found ==================')
print(grid_search.best_params_)

# Evaluate the best model found by GridSearchCV on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print('\n================== test accuracy ==================')
print(test_accuracy)

y_pred = best_model.predict(X_test)

# Print Confusion Matrix
print('\n================== confusion matrix ==================')
print(confusion_matrix(y_test, y_pred))

# Print Classification Report
print('\n================== classification report ==================')
print(classification_report(y_test, y_pred))

# Get the coefficients of the features from the logistic regression model within the pipeline
feature_importances = best_model.named_steps['logreg'].coef_[0]

# Print feature names and their coefficients
print('\n================== feature importance ==================')
for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance}")


##################
## Decision Tree
    
print('\n================== Decision Tree ==================')
print('\n======================================================')

# Create a pipeline that standardizes the data then creates a Decision Tree model
pipeline = Pipeline([
    ('dtree', DecisionTreeClassifier())  # Decision Tree classifier
])

# Define the parameter grid
param_grid = {
    'dtree__max_depth': np.arange(3, 16),  
    'dtree__min_samples_split': [2, 5, 10, 20],         
    'dtree__criterion': ['gini', 'entropy']             
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print('\n================== Best parameters found ==================')
print(grid_search.best_params_)

# Evaluate the best model found by GridSearchCV on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print('\n================== test accuracy ==================')
print(test_accuracy)
y_pred = best_model.predict(X_test)

# Print Confusion Matrix
print('\n================== confusion matrix ==================')
print(confusion_matrix(y_test, y_pred))

# Print Classification Report
print('\n================== classification report ==================')
print(classification_report(y_test, y_pred))

feature_importances = best_model.named_steps['dtree'].feature_importances_

# Print feature names and their importances
print('\n================== feature importance ==================')

for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance}")


##################
## Random Forest Classifier

print('\n================== Random Forest Classifier ==================')
print('\n======================================================')

# Create a pipeline that standardizes the data then creates a Random Forest Classifier model
pipeline = Pipeline([
    ('rf', RandomForestClassifier())
])

# Define the parameter grid
param_grid = {
    'rf__n_estimators': [100, 200, 300],  # Number of trees in the forest
    'rf__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'rf__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'rf__min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print('\n================== Best parameters found ==================')
print(grid_search.best_params_)

# Evaluate the best model found by GridSearchCV on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print('\n================== test accuracy ==================')
print(test_accuracy)

y_pred = best_model.predict(X_test)

# Print Confusion Matrix
print('\n================== confusion matrix ==================')
print(confusion_matrix(y_test, y_pred))

# Print Classification Report
print('\n================== classification report ==================')
print(classification_report(y_test, y_pred))

# Get the feature importances from the Random Forest model within the pipeline
feature_importances = best_model.named_steps['rf'].feature_importances_

# Print feature names and their importances
print('\n================== Feature Importance ==================')
for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance}")