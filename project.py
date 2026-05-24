import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):

    df = pd.read_csv("TelcoCustomerChurn.csv")

    df = df[[
        "TenureinMonths",
        "MonthlyCharge",
        "TotalCharges",
        "Contract",
        "InternetType",
        "PaymentMethod",
        "OnlineSecurity",
        "PremiumTechSupport",
        "PaperlessBilling",
        "ChurnLabel"
    ]]

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    X = df.drop("ChurnLabel", axis=1)

    y = df["ChurnLabel"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_test.to_csv(
        "input.csv",
        index=False
    )

    y_test.to_csv(
        "actual_output.csv",
        index=False
    )

    num_attribs = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    cat_attribs = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    pipeline = build_pipeline(
        num_attribs,
        cat_attribs
    )

    X_train_prepared = pipeline.fit_transform(X_train)

    X_test_prepared = pipeline.transform(X_test)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42
    )

    model.fit(
        X_train_prepared,
        y_train
    )

    y_pred = model.predict(X_test_prepared)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    output_df = X_test.copy()

    output_df["Actual_Churn"] = y_test.values

    output_df["Predicted_Churn"] = y_pred

    output_df.to_csv(
        "output.csv",
        index=False
    )

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        pipeline,
        PIPELINE_FILE
    )

    print("\nModel and Pipeline Saved")
    print("Output File Generated")

else:

    model = joblib.load(MODEL_FILE)

    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")

    transformed_input = pipeline.transform(
        input_data
    )

    predictions = model.predict(
        transformed_input
    )

    input_data["Predicted_Churn"] = predictions

    input_data.to_csv(
        "output.csv",
        index=False
    )

    print("\nPrediction Completed")