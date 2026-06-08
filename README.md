# CardioSense — Heart Disease Predictor

A machine learning web application built with **Flask** that predicts the likelihood of heart disease based on 13 clinical parameters. The app uses a trained **Random Forest classifier** on the UCI Heart Disease dataset and exposes both a user-friendly web interface and a REST API.

> **Academic Disclaimer:** This application is built for educational purposes as part of the Python Foundations for Data Science module. It is **not** a medical device and should **not** be used for actual clinical diagnosis.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Project Structure](#3-project-structure)
4. [Dataset Description](#4-dataset-description)
5. [Machine Learning Model](#5-machine-learning-model)
6. [Installation & Setup](#6-installation--setup)
7. [Running the Application](#7-running-the-application)
8. [Application Routes & API Reference](#8-application-routes--api-reference)
9. [Database](#9-database)
10. [Docker Deployment](#10-docker-deployment)
11. [Running Tests](#11-running-tests)
12. [Technology Stack](#12-technology-stack)
13. [PEP 8 & Code Quality](#13-pep-8--code-quality)
14. [Team & Contributions](#14-team--contributions)
15. [AI Tools Used](#15-ai-tools-used)

---

## 1. Project Overview

**CardioSense** is a full-stack Python web application that allows users to enter their clinical measurements and instantly receive a heart disease risk prediction from a pre-trained machine learning model.

The application has three main layers:

| Layer | Technology | Role |
|---|---|---|
| Front-end | HTML5, CSS3, JavaScript | User interface for input and results |
| Back-end | Python, Flask | Business logic, routing, API |
| Data layer | SQLite via SQLAlchemy | Stores prediction history |

### Why Heart Disease?

Heart disease is the leading cause of death globally (WHO, 2023). Early detection using clinical data can significantly improve patient outcomes. This project demonstrates how machine learning can be applied to real-world medical data to build a decision-support tool.

---

## 2. Features

### User-Facing Features
- **Prediction Form** — Enter 13 clinical parameters through a clean, validated web form
- **Instant Results** — See prediction result (Heart Disease / No Heart Disease) with a confidence percentage and animated progress bar
- **Patient Records Explorer** — Browse the full 1,024-row UCI dataset in a searchable table with Disease/Healthy badges
- **Responsive Design** — Works on desktop and mobile screens

### Technical Features
- **REST API** — JSON endpoints for programmatic access (predict, history, health check, dataset)
- **Prediction Logging** — Every prediction is saved to a SQLite database with timestamp
- **Model Confidence** — The app returns not just a prediction but the model's probability score
- **Docker Support** — Containerised with Docker and Docker Compose for easy deployment
- **Test Suite** — Automated tests using pytest covering all major routes
- **Environment Config** — Secrets managed via environment variables, not hardcoded

---

## 3. Project Structure

```
heart-disease-predictor/
│
├── app/                        # Main Flask application package
│   ├── __init__.py             # App factory — creates and configures Flask app
│   ├── models.py               # SQLAlchemy database models
│   ├── routes.py               # All URL routes and API endpoints
│   └── templates/              # Jinja2 HTML templates
│       ├── index.html          # Home page — prediction input form
│       ├── result.html         # Prediction result display page
│       └── data.html           # Dataset explorer page
│
├── data/
│   └── heart.csv               # UCI Heart Disease dataset (1,024 records)
│
├── model/
│   └── tuned_best_classifier.pkl   # Trained & serialised Random Forest model
│
├── instance/
│   └── predictions.db          # SQLite database (auto-created on first run)
│
├── tests/
│   └── test_routes.py          # pytest test suite for all routes
│
├── venv/                       # Python virtual environment (not committed to Git)
│
├── run.py                      # Application entry point (development server)
├── requirements.txt            # All Python dependencies with pinned versions
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose service configuration
├── .env.example                # Template for environment variables
├── .gitignore                  # Files and folders excluded from Git
└── README.md                   # This file
```

---

## 4. Dataset Description

**File:** `data/heart.csv`
**Source:** UCI Machine Learning Repository — Cleveland Heart Disease Dataset
**Size:** 1,024 patient records, 15 columns

### Feature Descriptions

| Column | Full Name | Type | Description |
|---|---|---|---|
| `age` | Age | Numeric | Patient age in years |
| `sex` | Sex | Binary | 1 = Male, 0 = Female |
| `cp` | Chest Pain Type | Categorical | 1=Typical angina, 2=Atypical angina, 3=Non-anginal pain, 4=Asymptomatic |
| `trestbps` | Resting Blood Pressure | Numeric | In mm Hg at hospital admission |
| `chol` | Serum Cholesterol | Numeric | In mg/dl |
| `fbs` | Fasting Blood Sugar | Binary | 1 = >120 mg/dl, 0 = ≤120 mg/dl |
| `restecg` | Resting ECG Results | Categorical | 0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy |
| `thalach` | Max Heart Rate Achieved | Numeric | Maximum heart rate during exercise test |
| `exang` | Exercise Induced Angina | Binary | 1 = Yes, 0 = No |
| `oldpeak` | ST Depression | Numeric | ST depression induced by exercise relative to rest |
| `slope` | Slope of ST Segment | Categorical | 1=Upsloping, 2=Flat, 3=Downsloping |
| `ca` | Number of Major Vessels | Numeric | 0-3 vessels coloured by fluoroscopy |
| `thal` | Thalassemia | Categorical | 3=Normal, 6=Fixed defect, 7=Reversible defect |
| `num` | Original Target | Numeric | Diagnosis of heart disease (0-4) |
| `target_binary` | Binary Target | Binary | **0 = No Disease, 1 = Heart Disease** (used for training) |

### Dataset Statistics

| Statistic | Value |
|---|---|
| Total Records | 1,024 |
| Heart Disease Cases | 470 (45.9%) |
| No Disease Cases | 554 (54.1%) |
| Missing Values | None (pre-cleaned) |
| Features used for prediction | 13 |

---

## 5. Machine Learning Model

### Algorithm: Random Forest Classifier

A **Random Forest** is an ensemble learning method that builds multiple decision trees during training and outputs the majority vote for classification.

**Why Random Forest?**
- Handles mixed data types (numeric + categorical) well
- Robust to outliers — common in medical data
- Provides probability scores, not just binary predictions
- Less prone to overfitting compared to a single decision tree

### How Prediction Works

1. User enters 13 values in the form
2. Flask reads the form data and converts to a Python list of floats
3. The list is reshaped to a NumPy 2D array: `shape = (1, 13)`
4. `model.predict()` returns `0` or `1`
5. `model.predict_proba()` returns probability for each class
6. The confidence shown is the probability of the predicted class

```python
prediction = int(model.predict(input_array)[0])
probability = model.predict_proba(input_array)[0][prediction] * 100
```

---

## 6. Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip
- Git
- Docker (optional)

### Step-by-Step Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/PMKUSUMA/heart-disease-predictor.git
cd heart-disease-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and set SECRET_KEY
```

---

## 7. Running the Application

```bash
python run.py
```

Open browser at: **http://127.0.0.1:5000**

| URL | Page |
|---|---|
| `http://127.0.0.1:5000/` | Prediction form |
| `http://127.0.0.1:5000/data` | Patient Records explorer |
| `http://127.0.0.1:5000/api/health` | API health check |
| `http://127.0.0.1:5000/api/predict` | JSON prediction endpoint |
| `http://127.0.0.1:5000/api/history` | Last 20 predictions |
| `http://127.0.0.1:5000/api/data` | Full dataset as JSON |

---

## 8. Application Routes & API Reference

### `GET /`
Renders the prediction input form.

### `POST /predict`
Accepts form submission, returns result page with prediction and confidence.

### `GET /data`
Renders the Patient Records explorer — full dataset in a searchable table.

### `GET /api/health`
```json
{ "status": "ok", "model": "Heart Disease Predictor" }
```

### `POST /api/predict`
**Request:**
```json
{
  "age": 54, "sex": 1, "cp": 0, "trestbps": 122,
  "chol": 286, "fbs": 0, "restecg": 0, "thalach": 116,
  "exang": 1, "oldpeak": 3.2, "slope": 1, "ca": 2, "thal": 2
}
```
**Response:**
```json
{ "prediction": 0, "label": "No Heart Disease", "confidence": 82.28 }
```

### `GET /api/history`
Returns last 20 predictions from the database as JSON.

### `GET /api/data`
Returns full dataset: `{ "count": 1024, "columns": [...], "data": [...] }`

---

## 9. Database

Uses **SQLite** via **Flask-SQLAlchemy**.

### PredictionLog Table

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Auto-incremented |
| `age` | Float | Patient age |
| `sex` | Float | Patient sex |
| `prediction` | Integer | 0 or 1 |
| `confidence` | Float | Probability score (0-100) |
| `timestamp` | DateTime | Auto-set on insert |

Database file: `instance/predictions.db` (auto-created, excluded from Git)

---

## 10. Docker Deployment

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up --build -d

# Stop
docker-compose down
```

App available at: **http://localhost:5000**

The `Dockerfile` uses `python:3.10-slim` and runs the app via **gunicorn** (production WSGI server). A Docker volume persists the SQLite database across restarts.

---

## 11. Running Tests

```bash
# Activate venv first, then:
pytest tests/ -v
```

| Test | What It Checks |
|---|---|
| `test_index_page` | GET / returns 200 and contains "CardioSense" |
| `test_health_api` | GET /api/health returns `{"status": "ok"}` |
| `test_api_predict_missing_data` | POST with empty JSON returns 400 |
| `test_api_predict_valid` | Valid prediction returns prediction + confidence |

---

## 12. Technology Stack

| Technology | Version | Purpose |
|---|---|---|
| Flask | 3.1.3 | Web framework |
| Flask-SQLAlchemy | 3.1.1 | Database ORM |
| scikit-learn | 1.6.1 | Random Forest model |
| NumPy | 2.2.6 | Array operations |
| Gunicorn | 26.0.0 | Production WSGI server |
| pytest | 9.0.3 | Testing framework |
| Docker | — | Containerisation |

---

## 13. PEP 8 & Code Quality

- **Type hints** on all variables and function signatures
- **snake_case** for variables/functions, **UPPER_SNAKE_CASE** for constants, **PascalCase** for classes
- **DRY principle** — model loaded once, reused across all requests
- **Separation of concerns** — models, routes, templates each in their own file
- **Environment variables** for secrets — no hardcoded keys
- **Error handling** on all prediction routes with appropriate HTTP status codes

---

## 14. Team & Contributions

| Student | GitHub | Contribution |
|---|---|---|
| Kusuma | PMKUSUMA | Flask app, templates, dataset, ML model, Docker, README (60%) |
| Praveen | Praveen2541 | Test suite, code review (40%) |

---

## 15. AI Tools Used

**Tool:** Claude (Anthropic)

**Prompts used:**
1. *"Check the heart-disease-predictor project and add all missing files — requirements.txt, .gitignore, Dockerfile, docker-compose.yml. Add a dataset viewing route so users can see the full CSV data."*
2. *"Run the Flask app and take screenshots of all pages."*
3. *"Create a detailed README.md explanation file for the project."*

---

*Built for Python Foundations for Data Science (k_ADSA_002) — 2026*
