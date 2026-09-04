# Spam Email/SMS Classifier

A production-ready machine learning application that classifies text messages as **Spam** or **Ham** using TF-IDF and Logistic Regression.

The project demonstrates an end-to-end machine learning lifecycle:

**Data → Preprocessing → Training → Evaluation → MLflow → Model Registry → Model Promotion → FastAPI → Testing → Docker → GitHub Actions → GHCR → Render**

---

## 🚀 Live Demo

### Frontend

https://spam-classifier-frontend-8z0o.onrender.com

### Backend API

https://spam-classifier-latest-1.onrender.com

### Swagger API Documentation

https://spam-classifier-latest-1.onrender.com/docs

### Health Check

https://spam-classifier-latest-1.onrender.com/health

---

## 📌 Project Overview

The application accepts an SMS or email message and predicts whether it is:

- `spam`
- `ham`

Example:

```text
Input:
Congratulations! You won a free prize!

Output:
spam