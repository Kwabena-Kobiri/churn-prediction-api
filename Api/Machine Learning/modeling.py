# Original file is located at
#     https://colab.research.google.com/drive/1iz8U4TNMIOYWgC8x1vNUW4hsdU0f6N1_
# """

import joblib
import numpy as np

# # Import some libraries
import pandas as pd

# Import Data
print("Loading data...")
train = pd.read_csv("../../../Data/Train.csv")
test = pd.read_csv("../../../Data/Test.csv")
submission = pd.read_csv("../../../Data/SampleSubmission.csv")

print(train.head())
print()
print("data has been loaded...")

print("Shape of train data: ", train.shape)
print("Shape of test data: ", test.shape)

# We will drop REGION, TOP_PACK, and MRG
# We will also replace the missing values for the numerical columns with their means (averages)

train.drop(columns=["REGION", "MRG", "TOP_PACK"], inplace=True)  # drop these columns
print()
print("AFTER REMOVING REGION, MRG and TOP_PACK")
print(train.head())

test.drop(columns=["REGION", "MRG", "TOP_PACK"], inplace=True)

# # Fill NAs for train data

train["MONTANT"].fillna((train["MONTANT"].mean()), inplace=True)
train["FREQUENCE_RECH"].fillna((train["FREQUENCE_RECH"].mean()), inplace=True)
train["REVENUE"].fillna((train["REVENUE"].mean()), inplace=True)
train["ARPU_SEGMENT"].fillna((train["ARPU_SEGMENT"].mean()), inplace=True)
train["FREQUENCE"].fillna((train["FREQUENCE"].mean()), inplace=True)
train["DATA_VOLUME"].fillna((train["DATA_VOLUME"].mean()), inplace=True)
train["ON_NET"].fillna((train["ON_NET"].mean()), inplace=True)
train["ORANGE"].fillna((train["ORANGE"].mean()), inplace=True)
train["TIGO"].fillna((train["TIGO"].mean()), inplace=True)
train["ZONE1"].fillna((train["ZONE1"].mean()), inplace=True)
train["ZONE2"].fillna((train["ZONE2"].mean()), inplace=True)
train["FREQ_TOP_PACK"].fillna((train["FREQ_TOP_PACK"].mean()), inplace=True)

# We Remove the ZONE1 and ZONE2 variables
train = train.drop(["ZONE1", "ZONE2"], axis=1)

# # Fill NAs for test data

test["MONTANT"].fillna((test["MONTANT"].mean()), inplace=True)
test["FREQUENCE_RECH"].fillna((test["FREQUENCE_RECH"].mean()), inplace=True)
test["REVENUE"].fillna((test["REVENUE"].mean()), inplace=True)
test["ARPU_SEGMENT"].fillna((test["ARPU_SEGMENT"].mean()), inplace=True)
test["FREQUENCE"].fillna((test["FREQUENCE"].mean()), inplace=True)
test["DATA_VOLUME"].fillna((test["DATA_VOLUME"].mean()), inplace=True)
test["ON_NET"].fillna((test["ON_NET"].mean()), inplace=True)
test["ORANGE"].fillna((test["ORANGE"].mean()), inplace=True)
test["TIGO"].fillna((test["TIGO"].mean()), inplace=True)
test["ZONE1"].fillna((test["ZONE1"].mean()), inplace=True)
test["ZONE2"].fillna((test["ZONE2"].mean()), inplace=True)
test["FREQ_TOP_PACK"].fillna((test["FREQ_TOP_PACK"].mean()), inplace=True)

# # Dropping ZONE1 and ZONE2 in the test dataset
test = test.drop(["ZONE1", "ZONE2"], axis=1)

print()
print("FINAL TRAIN DATA TO WORK WITH")
print(train.head())
print()
print("FINAL TEST DATA TO WORK WITH")
print(test.head())


# """## Machine Learning"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

dropcols = ["user_id", "CHURN"]
y = train["CHURN"]
x = train.drop(columns=dropcols, axis=1)
test = test.drop(
    columns=["user_id"], axis=1
)  # you will use this for predicting and submitting the result
print()
print("Shape of predictor variables: ", x.shape)
print("Shape of target variable: ", y.shape)
print("Shape of test data: ", test.shape)

# Split training data into train and test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)


# Further split X_train and y_train into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=1
)

# Standardize numeric columns
num_cols = [
    "MONTANT",
    "FREQUENCE_RECH",
    "REVENUE",
    "ARPU_SEGMENT",
    "FREQUENCE",
    "DATA_VOLUME",
    "ON_NET",
    "ORANGE",
    "TIGO",
    "REGULARITY",
    "FREQ_TOP_PACK",
]

scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_train.head()

X_train.select_dtypes(include=["int64", "float64"]).describe().T

X_test[num_cols] = scaler.transform(X_test[num_cols])

X_test.head()

test[num_cols] = scaler.transform(test[num_cols])

X_val[num_cols] = scaler.transform(X_val[num_cols])


# Encode the TENURE column
encoder = LabelEncoder()
X_train["TENURE"] = encoder.fit_transform(X_train["TENURE"])

X_test["TENURE"] = encoder.transform(X_test["TENURE"])

X_val["TENURE"] = encoder.transform(X_val["TENURE"])

test["TENURE"] = encoder.transform(test["TENURE"])

print()
print(X_train.select_dtypes(include=["int64", "float64"]).describe().T)

###################################################################

print()
print()
print(test.select_dtypes(include=["int64", "float64"]).describe().T)
print(test.info())


####################################################################


# # Building an XGBClassifier model
# from xgboost import XGBClassifier

# from sklearn.metrics import log_loss

# print()
# print("Shape of X_test: ", X_test.shape)
# print("Shape of the test data: ", test.shape)


# XGB = XGBClassifier(max_depth=6, n_estimators=200)
# XGB.fit(X_train, y_train)
# xgb_pred = XGB.predict_proba(X_test)

# # Printing the log loss value of the model
# print()
# print("Logloss")
# print(log_loss(y_test, xgb_pred))
# print()


# # Making predictions on test dataset with the model
# test["CHURN"] = XGB.predict_proba(test)[:, 1]
# print(test.head())


# # SAVING THE MODEL
# joblib.dump(XGB, "XGBClassifier.pkl")
# print()
# print("Model has been saved successfully")
