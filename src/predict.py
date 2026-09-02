import joblib

from src.preprocess import clean_text


MODEL_PATH = "models/spam_model.pkl"


model = joblib.load(MODEL_PATH)


def predict_message(message: str):

    cleaned_message = clean_text(message)

    prediction = model.predict(
        [cleaned_message]
    )[0]

    probabilities = model.predict_proba(
        [cleaned_message]
    )[0]

    classes = model.classes_

    spam_index = list(classes).index("spam")

    spam_probability = probabilities[spam_index]

    return prediction, spam_probability