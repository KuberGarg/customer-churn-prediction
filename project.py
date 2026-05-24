# CUSTOMER CHURN PREDICTION USING CATBOOST

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)

from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier


# FILE NAMES

MODEL_FILE = "model.pkl"

PIPELINE_FILE = "pipeline.pkl"


# BUILD PIPELINE

def build_pipeline(num_attribs, cat_attribs):

    num_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )
    ])

    cat_pipeline = Pipeline([
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    full_pipeline = ColumnTransformer([
        (
            "num",
            num_pipeline,
            num_attribs
        ),

        (
            "cat",
            cat_pipeline,
            cat_attribs
        )
    ])

    return full_pipeline


# TRAIN MODEL

if not os.path.exists(MODEL_FILE):


    # LOAD DATASET

    df = pd.read_csv(
        "TelcoCustomerChurn.csv"
    )


    # SELECT IMPORTANT COLUMNS

    df = df[[
        "TenureinMonths",
        "Contract",
        "MonthlyCharge",
        "TotalCharges",
        "InternetType",
        "PaymentMethod",
        "SatisfactionScore",
        "OnlineSecurity",
        "PremiumTechSupport",
        "ChurnLabel"
    ]]


    # REMOVE MISSING VALUES

    df.dropna(inplace=True)


    # LABEL ENCODING TARGET

    le = LabelEncoder()

    df["ChurnLabel"] = le.fit_transform(
        df["ChurnLabel"]
    )


    # FEATURES AND LABEL

    X = df.drop(
        "ChurnLabel",
        axis=1
    )

    y = df["ChurnLabel"]


    # NUMERICAL & CATEGORICAL COLUMNS

    num_attribs = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    cat_attribs = X.select_dtypes(
        include=["object"]
    ).columns.tolist()


    # TRAIN TEST SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # SAVE INPUT CSV

    X_test.to_csv(
        "input.csv",
        index=False
    )


    # BUILD PIPELINE

    pipeline = build_pipeline(
        num_attribs,
        cat_attribs
    )


    # DATA PREPROCESSING

    X_train_prepared = pipeline.fit_transform(
        X_train
    )

    X_test_prepared = pipeline.transform(
        X_test
    )


    # MODEL

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42
    )


    # TRAIN MODEL

    model.fit(
        X_train_prepared,
        y_train
    )


    # SAVE MODEL & PIPELINE

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        pipeline,
        PIPELINE_FILE
    )


    # PREDICTION

    y_pred = model.predict(
        X_test_prepared
    )


    # ACCURACY

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"\nModel Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        "\nModel and Pipeline Saved"
    )


# LOAD MODEL AND TEST

else:


    # LOAD MODEL & PIPELINE

    model = joblib.load(
        MODEL_FILE
    )

    pipeline = joblib.load(
        PIPELINE_FILE
    )


    # LOAD DATASET

    df = pd.read_csv(
        "TelcoCustomerChurn.csv"
    )


    # SELECT IMPORTANT COLUMNS

    df = df[[
        "TenureinMonths",
        "Contract",
        "MonthlyCharge",
        "TotalCharges",
        "InternetType",
        "PaymentMethod",
        "SatisfactionScore",
        "OnlineSecurity",
        "PremiumTechSupport",
        "ChurnLabel"
    ]]


    # REMOVE MISSING VALUES

    df.dropna(inplace=True)


    # LABEL ENCODING TARGET

    le = LabelEncoder()

    df["ChurnLabel"] = le.fit_transform(
        df["ChurnLabel"]
    )


    # FEATURES & LABEL

    X = df.drop(
        "ChurnLabel",
        axis=1
    )

    y = df["ChurnLabel"]


    # TRAIN TEST SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # LOAD INPUT DATA

    input_data = pd.read_csv(
        "input.csv"
    )


    # TRANSFORM INPUT DATA

    transformed_input = pipeline.transform(
        input_data
    )


    # PREDICTION

    predictions = model.predict(
        transformed_input
    )


    # SAVE OUTPUT CSV

    output_data = input_data.copy()

    output_data["Actual_Churn"] = y_test.values

    output_data["Predicted_Churn"] = predictions

    output_data.to_csv(
        "output.csv",
        index=False
    )


    # ACCURACY

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nModel Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        "\nOutput saved as output.csv"
    )