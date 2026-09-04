# 🚀 Prasanna AI Solutions — Spam Message Classifier

> An end-to-end AI/ML application that detects whether a text message is **Spam** or **Ham (Normal)** using Natural Language Processing and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-black)
![DagsHub](https://img.shields.io/badge/MLOps-DagsHub-orange)
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

### 📊 DagsHub MLOps Repository

👉 https://dagshub.com/adhavprasanna/spam-classifier

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
- MLflow experiment tracking
- DagsHub-hosted model registry
- Automated model promotion and rejection
- REST API development
- Automated testing
- Docker containerization
- GitHub Actions CI/CD
- GitHub Container Registry
- Cloud deployment
- Web frontend integration

---

## 🔄 End-to-End MLOps Workflow

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
MLflow Experiment Tracking
   ↓
DagsHub Model Registry
   ↓
Production Alias (@production)
   ↓
GitHub Actions CI/CD
   ↓
Automated Tests
   ↓
Model Promotion / Rejection
   ↓
If Promoted → Docker Build
   ↓
GitHub Container Registry
   ↓
Render Cloud Deployment
   ↓
Web Frontend