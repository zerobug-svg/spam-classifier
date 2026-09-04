import dagshub
import mlflow

from src.preprocess import clean_text


REGISTERED_MODEL_NAME = "SpamMessageClassifier"
PRODUCTION_ALIAS = "production"

DAGSHUB_OWNER = "adhavprasanna"
DAGSHUB_REPO = "spam-classifier"


# Connect MLflow to DagsHub
dagshub.init(
    repo_owner=DAGSHUB_OWNER,
    repo_name=DAGSHUB_REPO,
    mlflow=True
)


print(
    "MLflow Tracking URI:",
    mlflow.get_tracking_uri()
)


# Load the production model from DagsHub Model Registry
MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
)


print(
    "Loading production model:",
    MODEL_URI
)


# Load the native scikit-learn model
model = mlflow.sklearn.load_model(
    MODEL_URI
)


print(
    "Production model loaded successfully!"
)


def predict_message(message: str):

    cleaned_message = clean_text(
        message
    )

    prediction = model.predict(
        [cleaned_message]
    )[0]

    probabilities = model.predict_proba(
        [cleaned_message]
    )[0]

    classes = model.classes_

    spam_index = list(classes).index(
        "spam"
    )

    spam_probability = probabilities[
        spam_index
    ]

    return prediction, spam_probability