# Spam Message Classifier

An end-to-end Machine Learning project that classifies SMS/email messages as **Spam** or **Ham (Normal)**.

The project includes text preprocessing, TF-IDF feature extraction, Logistic Regression, FastAPI, Pytest testing, Docker containerization, and Git/GitHub version control.

---

## 🚀 Project Overview

This project takes a text message as input and predicts whether the message is:

- `spam` — unwanted/promotional/fraudulent message
- `ham` — normal message

Example:

```text
Input:
Congratulations! You won a free iPhone!

Output:
spam
---

## 🏗️ Architecture

The application follows this machine learning workflow:

```text
SMS/Email Dataset
       ↓
Text Cleaning
       ↓
TF-IDF Vectorization
       ↓
Logistic Regression
       ↓
Model Evaluation
       ↓
Saved ML Model
       ↓
FastAPI
       ↓
REST API
       ↓
Docker Container
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
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/zerobug-svg/spam-classifier.git
cd spam-classifier
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```
---

## 🤖 Train the Model

Before running the API, the machine learning model must be trained.

From the project root directory, run:

```powershell
python src/train.py
```

The training script performs the following steps:

1. Loads the spam/ham dataset.
2. Cleans the text messages.
3. Splits the dataset into training and testing data.
4. Converts text into numerical features using TF-IDF.
5. Trains a Logistic Regression classifier.
6. Evaluates the model.
7. Saves the trained model.

The trained model is saved as:

```text
models/spam_model.pkl
```

> **Note:** The trained `.pkl` model is generated locally and is excluded from GitHub using `.gitignore`.
---

## 🚀 Run the FastAPI Application

After training the model, start the FastAPI server.

From the project root directory, run:

```powershell
uvicorn api.main:app --reload
```

The API will start at:

```text
http://127.0.0.1:8000
```

### API Documentation

FastAPI automatically provides interactive API documentation.

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

From the Swagger UI, you can test the `/predict` endpoint directly.

### Health Check

The API provides a health-check endpoint:

```text
GET /health
```

Expected response:

```json
{
  "status": "healthy"
}
```
---

## 🧪 Test the Prediction API

The `/predict` endpoint accepts a message and returns whether it is classified as spam or ham.

### Endpoint

```text
POST /predict
```

### Example Request

```json
{
  "message": "Congratulations! You won a free iPhone!"
}
```

### Example Response

```json
{
  "message": "Congratulations! You won a free iPhone!",
  "prediction": "spam",
  "spam_probability": 0.6099
}
```

### Normal Message Example

```json
{
  "message": "Can you send me the project report?"
}
```

The API returns the predicted class and the probability that the message is spam.

### Testing Through Swagger UI

Open:

```text
http://127.0.0.1:8000/docs
```

Then:

1. Open the `POST /predict` endpoint.
2. Click **Try it out**.
3. Enter a message.
4. Click **Execute**.
5. Check the prediction returned by the API.
---

## 🧪 Automated Testing

This project uses **Pytest** to test the FastAPI application.

Run the tests from the project root directory:

```powershell
pytest
```

The test suite checks:

- API health check
- Spam message prediction
- Normal message prediction

### Expected Result

```text
3 passed
```

A successful test run confirms that the API endpoints are working correctly.
---

## 🐳 Run with Docker

Docker allows the application to run inside a container with all required dependencies.

### 1. Build the Docker Image

Make sure Docker Desktop is running.

From the project root directory, run:

```powershell
docker build -t spam-classifier-api .
```

This creates a Docker image named:

```text
spam-classifier-api
```

### 2. Run the Docker Container

Run:

```powershell
docker run -p 8000:8000 spam-classifier-api
```

The FastAPI application will now run inside the Docker container.

The API will be available at:

```text
http://localhost:8000
```

### 3. Open API Documentation

Open the following URL in your browser:

```text
http://localhost:8000/docs
```

You can use Swagger UI to test the spam classification API.

### 4. Stop the Container

To stop the running container, press:

```text
Ctrl + C
```

in the terminal where the container is running.

### Docker Architecture

```text
Docker Image
     ↓
Docker Container
     ↓
FastAPI Application
     ↓
Spam Classifier Model
     ↓
Prediction
```
---

## 🔄 Git and GitHub

Git is used for version control, while GitHub is used to store and share the project repository.

### Check Git Status

```powershell
git status
```

### Add Changes

```powershell
git add .
```

### Commit Changes

```powershell
git commit -m "Update project documentation"
```

### Push Changes to GitHub

```powershell
git push
```

The project repository is available on GitHub:

```text
https://github.com/zerobug-svg/spam-classifier
```
---

## 🔮 Future Improvements

Possible improvements for this project include:

- Use a larger real-world SMS/email dataset.
- Improve model accuracy with additional preprocessing.
- Compare multiple machine learning algorithms.
- Add confidence thresholds for predictions.
- Add a web-based user interface.
- Deploy the API to a cloud platform.
- Add CI/CD using GitHub Actions.
- Add API authentication and security.
- Add monitoring and logging.
- Add model versioning and experiment tracking.
---

## 👤 Author

**Prasanna Adhav**

AI/ML Engineer

---

## 📄 License

This project is created for learning and demonstration purposes.