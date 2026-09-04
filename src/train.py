import os
from datetime import datetime

import pandas as pd
import joblib
import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.preprocess import clean_text


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "data/spam.csv"
MODEL_PATH = "models/spam_model.pkl"

MODEL_VERSION = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

REGISTERED_MODEL_NAME = "SpamMessageClassifier"


# ==========================================
# DagsHub MLflow CONNECTION
# ==========================================

dagshub.init(
    repo_owner="adhavprasanna",
    repo_name="spam-classifier",
    mlflow=True
)

print(
    "MLflow Tracking URI:",
    mlflow.get_tracking_uri()
)


# ==========================================
# MLFLOW EXPERIMENT
# ==========================================

mlflow.set_experiment(
    "Spam Message Classifier"
)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    DATA_PATH
)

print(
    "Dataset loaded:",
    df.shape
)


# ==========================================
# TEXT PREPROCESSING
# ==========================================

df["clean_message"] = df["message"].apply(
    clean_text
)

X = df["clean_message"]
y = df["label"]


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ==========================================
# START MLFLOW RUN
# ==========================================

mlflow.start_run()


try:

    # ======================================
    # MACHINE LEARNING PIPELINE
    # ======================================

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer()
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced"
            )
        )
    ])


    # ======================================
    # TRAIN MODEL
    # ======================================

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete!"
    )


    # ======================================
    # MODEL PREDICTION
    # ======================================

    y_pred = model.predict(
        X_test
    )


    # ======================================
    # MODEL EVALUATION
    # ======================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="spam",
        zero_division=0
    )


    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")

    print(
        f"Accuracy : {accuracy:.2f}"
    )

    print(
        f"Precision: {precision:.2f}"
    )

    print(
        f"Recall   : {recall:.2f}"
    )

    print(
        f"F1 Score : {f1:.2f}"
    )


    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    # ======================================
    # LOG PARAMETERS TO MLFLOW
    # ======================================

    mlflow.log_param(
        "model_version",
        MODEL_VERSION
    )

    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "class_weight",
        "balanced"
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "registered_model_name",
        REGISTERED_MODEL_NAME
    )


    # ======================================
    # LOG METRICS TO MLFLOW
    # ======================================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )


    # ======================================
    # LOG MODEL TO MLFLOW MODEL REGISTRY
    # ======================================

    mlflow.sklearn.log_model(
        sk_model=model,
        name="spam_classifier_model",
        registered_model_name=REGISTERED_MODEL_NAME
    )


    # ======================================
    # SAVE MODEL LOCALLY
    # ======================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        "\nModel saved successfully!"
    )

    print(
        f"Location: {MODEL_PATH}"
    )

    print(
        f"Model Version: {MODEL_VERSION}"
    )

    print(
        f"Registered Model: {REGISTERED_MODEL_NAME}"
    )


finally:

    # ======================================
    # END MLFLOW RUN
    # ======================================

    mlflow.end_run()