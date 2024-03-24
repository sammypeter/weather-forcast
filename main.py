# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

data = 'Weather Training Data.csv'
df = pd.read_csv(data) 

# Find all categorical variables
categorical = df.select_dtypes(include=['object']).columns.tolist()

# Count the binary categorical variables
binary_categorical = [col for col in categorical if df[col].nunique() == 2] 

# Check for missing values in categorical variables
missing_categorical = df[categorical].isnull().sum()

# Filter the categorical variables with missing values
categorical_with_missing = missing_categorical[missing_categorical > 0] 

# Find all numerical variables by excluding object (categorical) variables
numerical = df.select_dtypes(exclude=['object']).columns.tolist() 

# Initialize a dictionary to store potential outliers
potential_outliers = {}

# Function to identify and plot outliers for numerical columns
for col in numerical:
    # Calculate Q1 and Q3 values
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define lower and upper bounds for potential outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify and store potential outliers
    potential_outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)] 

# Print the numerical columns containing potential outliers 
for col, data in potential_outliers.items():
    if not data.empty:
        col

#Remove outliers
numerical.remove('RainTomorrow')
for col in numerical:
    # Calculate Q1 and Q3 values
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define lower and upper bounds for potential outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Replace missing values in numerical columns with the median
for col in numerical:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)
# Replace missing values in categorical columns with the mode (most frequent value)
for col in categorical:
    mode_value = df[col].mode()[0]
    df[col].fillna(mode_value, inplace=True)

# Create 'RainToday_0' and 'RainToday_1' columns based on 'RainToday' values
df['RainToday_0'] = (df['RainToday'] == 'No').astype(int)
df['RainToday_1'] = (df['RainToday'] == 'Yes').astype(int)

X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
X_train.shape, X_test.shape

#Importing the LabelEncoder class from scikit-learn.
from sklearn.preprocessing import LabelEncoder
#Converting the target variable y_train and y_test to DataFrames.
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
#Filling missing values in the 'RainTomorrow' column with the mode (most frequent value) of that column.
y_train['RainTomorrow'].fillna(y_train['RainTomorrow'].mode()[0], inplace=True)
y_test['RainTomorrow'].fillna(y_test['RainTomorrow'].mode()[0], inplace=True)
#Creating two instances of the LabelEncoder class, one for the training data and one for the test data.
train_labelled = LabelEncoder()
test_labelled = LabelEncoder()
#Fitting the LabelEncoder to the unique values in the 'RainTomorrow' column
train_labelled.fit(y_train['RainTomorrow'].astype('str').drop_duplicates())
test_labelled.fit(y_test['RainTomorrow'].astype('str').drop_duplicates())
#Replaces the original categorical values with their corresponding numerical encodings.
y_train['enc'] = train_labelled.transform(y_train['RainTomorrow'].astype('str'))
y_test['enc'] = train_labelled.transform(y_test['RainTomorrow'].astype('str'))
#Dropping the original 'RainTomorrow' column from both the training and test DataFrames since it's no longer needed
y_train.drop(columns=['RainTomorrow'], inplace=True)
y_test.drop(columns=['RainTomorrow'], inplace=True) 

X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location),
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)

X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location),
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
# Predict on the training data
train_predictions = model.predict(X_train)

# Predict on the test data
test_predictions = model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
confusion = confusion_matrix(y_test, test_predictions)

from sklearn.metrics import classification_report
report = classification_report(y_test, test_predictions) 

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)