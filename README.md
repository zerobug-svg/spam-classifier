# 🚀 Prasanna AI Solutions — Spam Message Classifier

> An end-to-end AI/ML application that detects whether a text message is **Spam** or **Ham (Normal)** using Natural Language Processing and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-black)
![Render](https://img.shields.io/badge/Deployed%20on-Render-purple)

---

## 🌐 Live Demo

### 🖥️ Web Application

👉 https://spam-classifier-frontend-8z0o.onrender.com

### 🔧 Backend API

👉 https://spam-classifier-latest-1.onrender.com

### 📚 Swagger API Documentation

👉 https://spam-classifier-latest-1.onrender.com/docs

### ❤️ Health Check

👉 https://spam-classifier-latest-1.onrender.com/health

---

## 📌 Project Overview

**Prasanna AI Solutions — Spam Message Classifier** is an end-to-end Machine Learning application designed to classify text messages as:

- 🚨 **Spam**
- ✅ **Ham / Normal Message**

The project demonstrates the complete lifecycle of an AI/ML application, including:

- Data loading
- Text preprocessing
- TF-IDF feature extraction
- Machine Learning model training
- Model evaluation
- Model serialization
- REST API development
- Automated testing
- Docker containerization
- GitHub Actions CI/CD
- GitHub Container Registry
- Cloud deployment
- Web frontend integration

---

## 🔄 End-to-End Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
FastAPI
   ↓
Pytest
   ↓
Docker
   ↓
GitHub Actions
   ↓
GitHub Container Registry
   ↓
Render Cloud Deployment
   ↓
Web Frontend
```

---

## 🏗️ System Architecture

![Spam Message Classifier Architecture](images/architecture.png)

### Architecture Flow

```text
User
  ↓
Web Frontend
  ↓
FastAPI REST API
  ↓
Text Preprocessing
  ↓
TF-IDF Vectorization
  ↓
Logistic Regression
  ↓
Spam / Ham Prediction
  ↓
Prediction + Spam Probability
```

---

## 🤖 Machine Learning Approach

### 1. Data Preprocessing

The incoming text is cleaned before being passed to the Machine Learning model.

Processing includes:

- Converting text to lowercase
- Removing unnecessary special characters
- Removing extra spaces
- Preparing normalized text

Example:

```text
Original:
Congratulations! You WON ₹50,000!!!

Cleaned:
congratulations you won 50000
```

---

### 2. TF-IDF Feature Extraction

TF-IDF stands for:

**Term Frequency — Inverse Document Frequency**

It converts text into numerical features that can be understood by a Machine Learning model.

The pipeline uses:

```python
TfidfVectorizer()
```

---

### 3. Classification Model

The project uses:

```text
Logistic Regression
```

The complete Machine Learning pipeline is:

```text
Text
 ↓
Cleaning
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
Spam / Ham
```

---

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report

Example evaluation output from the development dataset:

```text
Accuracy : 1.00
Precision: 1.00
Recall   : 1.00
F1 Score : 1.00
```

> ⚠️ Note: The development dataset contains only 20 messages, so these metrics should not be interpreted as production-level performance. A much larger real-world dataset would be required for reliable evaluation.

---

## 🔌 REST API

The backend is built using **FastAPI**.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/predict` | Predict spam or ham |

---

## 🩺 Health Check

Request:

```text
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

## 🔮 Prediction API

Request:

```text
POST /predict
```

JSON:

```json
{
  "message": "Congratulations! You won a free iPhone!"
}
```

Example response:

```json
{
  "message": "Congratulations! You won a free iPhone!",
  "prediction": "spam",
  "spam_probability": 0.6099
}
```

---

## 🧪 Testing

The project uses **Pytest** for API testing.

Tests cover:

- Health endpoint
- Spam prediction
- Normal message prediction

Run tests:

```bash
pytest
```

Expected result:

```text
3 passed
```

---

## 🐳 Docker

The application is containerized using Docker.

### Build Docker Image

```bash
docker build -t spam-classifier-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 spam-classifier-api
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 🔄 CI/CD Pipeline

GitHub Actions automatically performs the following steps:

```text
Git Push
   ↓
GitHub Actions
   ↓
Install Python Dependencies
   ↓
Train Model
   ↓
Run Pytest
   ↓
Build Docker Image
   ↓
Login to GHCR
   ↓
Push Docker Image
```

The workflow is located at:

```text
.github/workflows/ci.yml
```

---

## 📦 GitHub Container Registry

The Docker image is published to GitHub Container Registry.

Image:

```text
ghcr.io/zerobug-svg/spam-classifier:latest
```

Pull the image:

```bash
docker pull ghcr.io/zerobug-svg/spam-classifier:latest
```

Run it:

```bash
docker run -p 8001:8000 ghcr.io/zerobug-svg/spam-classifier:latest
```

---

## ☁️ Cloud Deployment

The application is deployed using **Render**.

### Frontend

```text
https://spam-classifier-frontend-8z0o.onrender.com
```

### Backend

```text
https://spam-classifier-latest-1.onrender.com
```

### Swagger

```text
https://spam-classifier-latest-1.onrender.com/docs
```

### Health Check

```text
https://spam-classifier-latest-1.onrender.com/health
```

---

## 🌐 Frontend + Backend Connection

The frontend communicates with the FastAPI backend using the `/predict` endpoint.

```text
Browser
   ↓
Frontend
   ↓
HTTP POST Request
   ↓
FastAPI
   ↓
ML Model
   ↓
Prediction
   ↓
JSON Response
   ↓
Frontend
```

The frontend sends:

```json
{
  "message": "Congratulations! You won a free prize!"
}
```

The backend returns:

```json
{
  "prediction": "spam",
  "spam_probability": 0.61
}
```

---

## 📁 Project Structure

```text
spam-classifier/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── spam.csv
│
├── frontend/
│   └── index.html
│
├── images/
│   └── architecture.png
│
├── models/
│   └── spam_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

### Programming

- Python 3.11

### Machine Learning

- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Testing

- Pytest
- HTTPX

### DevOps

- Git
- GitHub
- Docker
- GitHub Actions
- GitHub Container Registry

### Cloud

- Render

### Frontend

- HTML
- CSS
- JavaScript
- Fetch API

---

## ⚙️ Local Installation

### Step 1 — Clone Repository

```bash
git clone https://github.com/zerobug-svg/spam-classifier.git
```

### Step 2 — Enter Project

```bash
cd spam-classifier
```

### Step 3 — Create Virtual Environment

Windows:

```powershell
python -m venv .venv
```

### Step 4 — Activate Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5 — Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 🧠 Train the Model

Run:

```powershell
python src/train.py
```

The trained model will be saved as:

```text
models/spam_model.pkl
```

---

## 🚀 Run the API Locally

Run:

```powershell
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## 🧪 Run Tests Locally

```powershell
pytest
```

---

## 🖥️ Web Application

The frontend provides a simple interface where users can enter a message and receive:

```text
Spam / Ham
+
Spam Probability
```

Example:

```text
Input:
Congratulations! You won a free iPhone!

Output:
🚨 SPAM

Spam probability: 60.99%
```

---

## 🔐 Model File

The trained model is generated locally and during CI/CD.

The model file:

```text
models/spam_model.pkl
```

is excluded from Git using `.gitignore`.

The Docker build includes the locally generated model when building the application image.

---

## 📈 Future Improvements

The project can be extended with:

- Larger real-world SMS dataset
- Email classification
- Better NLP preprocessing
- N-grams
- Hyperparameter tuning
- Cross-validation
- Multiple ML algorithms
- Random Forest
- SVM
- XGBoost
- Transformer-based models
- BERT
- Model monitoring
- Data drift detection
- MLflow experiment tracking
- Automated model retraining
- Database integration
- User authentication
- Production monitoring
- Kubernetes deployment
- AWS / Azure deployment

---

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

### Machine Learning

- NLP preprocessing
- Feature engineering
- TF-IDF
- Logistic Regression
- Model evaluation
- Model serialization

### Software Engineering

- Python project structure
- REST API development
- Unit/API testing
- Dependency management

### MLOps

- Model training pipeline
- Automated testing
- Docker
- CI/CD
- Container Registry
- Cloud deployment

### DevOps

- Git
- GitHub
- GitHub Actions
- Docker
- Deployment automation

---

## 👨‍💻 Author

### Prasanna Dattu Adhav

**AI & ML Engineer**

### Prasanna AI Solutions

Building practical AI/ML applications and end-to-end Machine Learning solutions.

---

## 📬 Contact

📧 Email:

adhavprasamna@gmail.com

📸 Instagram:

https://www.instagram.com/prasanna_045_ig/

🐙 GitHub:

https://github.com/zerobug-svg

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is created for learning, portfolio, and demonstration purposes.