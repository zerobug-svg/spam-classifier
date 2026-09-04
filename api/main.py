from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.predict import predict_message


app = FastAPI(
    title="Spam Message Classifier API",
    description="API for detecting spam messages",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {
        "message": "Spam Message Classifier API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: MessageRequest):

    prediction, probability, model_version = predict_message(
        request.message
    )

    return {
        "message": request.message,
        "prediction": prediction,
        "spam_probability": round(
            float(probability),
            4
        ),
        "model_version": str(model_version)
    }