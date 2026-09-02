from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_message


# Create FastAPI application
app = FastAPI(
    title="Spam Message Classifier",
    description="API for detecting spam messages",
    version="1.0.0"
)


# Input format
class MessageRequest(BaseModel):
    message: str


# Health check
@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: MessageRequest):

    prediction, probability = predict_message(
        request.message
    )

    return {
        "message": request.message,
        "prediction": prediction,
        "spam_probability": round(
            float(probability),
            4
        )
    }