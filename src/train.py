import os
import pandas as pd
import joblib

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

from preprocess import clean_text


DATA_PATH = "data/spam.csv"
MODEL_PATH = "models/spam_model.pkl"


# -----------------------------
# 1. Load dataset
# -----------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)


# -----------------------------
# 2. Clean messages
# -----------------------------

df["clean_message"] = df["message"].apply(clean_text)


# -----------------------------
# 3. Separate input and target
# -----------------------------

X = df["clean_message"]
y = df["label"]


# -----------------------------
# 4. Split data
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# 5. Create ML pipeline
# -----------------------------

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(class_weight="balanced"))
])

# -----------------------------
# 6. Train
# -----------------------------

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# -----------------------------
# 7. Predict
# -----------------------------

y_pred = model.predict(X_test)


# -----------------------------
# 8. Calculate metrics
# -----------------------------

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


# -----------------------------
# 9. Display results
# -----------------------------

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# -----------------------------
# 10. Save complete pipeline
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved successfully!")
print(f"Location: {MODEL_PATH}")