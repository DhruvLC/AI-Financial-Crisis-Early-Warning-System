# 🚨 AI Financial Crisis Early Warning System

> **An enterprise-grade AI platform for predicting financial distress using Machine Learning, Deep Learning, Transformer Models, and Self-Supervised Learning with a production-ready FastAPI backend and Streamlit frontend.**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub_Actions-success.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

# 📑 Table of Contents

- Overview
- Key Features
- System Architecture
- Technology Stack
- Machine Learning Pipeline
- Project Structure
- API Endpoints
- Installation
- Running the Application
- Docker
- Testing
- Results
- Future Improvements
- Author
- License

---

# 📌 Overview

Financial crises can significantly impact businesses, investors, and financial institutions. Early identification of financial distress enables proactive decision-making and risk mitigation.

This project delivers an end-to-end AI solution capable of predicting financial crisis risk through a production-ready architecture integrating:

- Traditional Machine Learning
- Deep Learning
- Transformer Models
- Self-Supervised Learning
- FastAPI Backend
- Streamlit Dashboard
- Docker Deployment
- CI/CD Pipeline

---

# ✨ Key Features

### Data Engineering

- Automated data ingestion
- Data validation
- Missing value handling
- Duplicate detection
- Outlier treatment
- Feature preprocessing

### Feature Engineering

- Automated feature generation
- Feature metadata management
- Feature validation
- Feature registry

### Machine Learning

- Multiple ML algorithms
- Hyperparameter tuning
- Model comparison
- Model selection
- Model Registry

### Deep Learning

- Neural Network models
- Performance evaluation
- Checkpoint management

### Transformer Models

- Financial feature transformer
- Advanced representation learning

### Self-Supervised Learning

- Representation learning
- Contrastive learning
- Feature embedding generation

### Backend

- FastAPI REST API
- Swagger UI
- ReDoc
- Batch prediction
- Model versioning
- Health monitoring

### Frontend

- Streamlit Dashboard
- Single Prediction
- Batch Prediction
- Model Information
- API Status
- Interactive Charts

### DevOps

- Docker support
- GitHub Actions
- Logging
- Monitoring
- Production configuration

---

# 🏗️ System Architecture

```text
                +------------------------+
                |     Streamlit UI       |
                +-----------+------------+
                            |
                            v
                +------------------------+
                |      FastAPI API       |
                +-----------+------------+
                            |
        +-------------------+-------------------+
        |                                       |
        v                                       v
 Data Validation                     Model Registry
        |                                       |
        +-------------------+-------------------+
                            |
                            v
                 Feature Engineering
                            |
                            v
                 Inference Pipeline
                            |
                            v
                  Prediction Results
```

---

# 🛠 Technology Stack

## Programming

- Python

## Machine Learning

- Scikit-learn
- XGBoost
- Pandas
- NumPy

## Deep Learning

- PyTorch *(or TensorFlow, depending on your implementation)*

## API

- FastAPI
- Uvicorn
- Pydantic

## Frontend

- Streamlit
- Plotly

## DevOps

- Docker
- Docker Compose
- GitHub Actions

---

# 🤖 Machine Learning Pipeline

```
Data Collection
        │
        ▼
Data Validation
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning
        │
        ▼
Deep Learning
        │
        ▼
Transformer Models
        │
        ▼
Self-Supervised Learning
        │
        ▼
Model Registry
        │
        ▼
FastAPI Backend
        │
        ▼
Streamlit Frontend
```

---

# 📂 Project Structure

```text
financial-crisis-ews/

├── src/
│   ├── api/
│   ├── pipeline/
│   ├── models/
│   ├── registry/
│   └── config/
│
├── frontend/
│   ├── pages/
│   ├── components/
│   ├── services/
│   └── app.py
│
├── reports/
│
├── tests/
│
├── docs/
│
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---

# 🌐 API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | / | Root |
| GET | /health | Health Check |
| GET | /version | API Version |
| GET | /metrics | Service Metrics |
| GET | /models | Active Model |
| POST | /predict | Single Prediction |
| POST | /predict/batch | Batch Prediction |
| POST | /validate | Input Validation |

Swagger UI

```
http://localhost:8000/docs
```

ReDoc

```
http://localhost:8000/redoc
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/DhruvLC/AI-Financial-Crisis-Early-Warning-System
cd AI-Financial-Crisis-Early-Warning-System
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run FastAPI

```bash
uvicorn src.api.app:app --reload
```

Backend

```
http://localhost:8000
```

---

# ▶️ Run Streamlit

```bash
streamlit run frontend/app.py
```

---

# 🐳 Docker

Build

```bash
docker compose up --build
```

---

# 🧪 Testing

Run tests

```bash
pytest
```

---

# 📈 Results

✔ End-to-End AI Pipeline

✔ Production Ready FastAPI Backend

✔ Interactive Streamlit Dashboard

✔ Docker Support

✔ GitHub Actions CI/CD

✔ Model Registry

✔ Automated Validation

✔ Batch Prediction

✔ Production Readiness: **9.0/10**

---

# 📸 Screenshots

Add screenshots after deployment.

Example:

```
Dashboard

Single Prediction

Batch Prediction

Model Information

API Status
```

---

# 🔮 Future Improvements

- Authentication & Authorization
- Prometheus & Grafana Monitoring
- Kubernetes Deployment
- Model Drift Detection
- Explainable AI (SHAP/LIME)
- Real-time Streaming Predictions

---

# 👨‍💻 Author

**Dhruv Patil**

- LinkedIn: https://www.linkedin.com/in/dhruv-patil-833b4b292/
- GitHub: https://github.com/DhruvLC

---

# 📄 License

This project is licensed under the MIT License.
