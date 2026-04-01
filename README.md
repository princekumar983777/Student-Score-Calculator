# 🎓 Student Exam Score Prediction (Production ML System)

This repository contains a **production-grade ML system** that predicts `exam_score` from tabular student data, trains **multiple models**, and **automatically selects the best** based on validation \(highest R²\).

It also includes a legacy Flask demo app (`app.py`) that can continue to be used independently.

## 📦 Project structure

```
.
├── artifacts/
│   ├── logs/
│   ├── metrics.json
│   ├── models/
│   │   ├── best_model.pkl
│   │   └── best_model.pt
│   └── scaler/
│       └── preprocessor.pkl
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── students.csv
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   └── utils/
├── tests/
├── dvc.yaml
├── params.yaml
├── main.py
└── requirements.txt
```

## ✅ Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run (CLI)

- **Ingest raw data** (copies from `research/` if needed):

```bash
python main.py ingest
```

- **Preprocess** (split train/val/test, fit & save preprocessor):

```bash
python main.py preprocess
```

- **Train + auto-select best model** (logs runs to MLflow, saves best model locally):

```bash
python main.py train
```

- **Evaluate** (prints `artifacts/metrics.json`):

```bash
python main.py evaluate
```

- **Predict** from a CSV:

```bash
python main.py predict --input data/raw/students.csv --output artifacts/predictions.csv
```

## 🌐 Run as Server (Automated ML Service)

The system can now run as a **continuous learning server** that automatically retrains when data changes:

### **Start Server with Initialization**
```bash
# Initialize system and start server
python start_server.py

# Or with custom options
python start_server.py --host 127.0.0.1 --port 8080
```

### **Server Features**
- **🔄 Auto-retraining**: Monitors data changes every 5 minutes and retrains automatically
- **📡 REST API**: FastAPI endpoints for predictions
- **📊 Batch predictions**: Process multiple students at once
- **📤 Data upload**: Upload new training data via API
- **🏥 Health checks**: Monitor server status
- **📈 Training status**: Check retraining progress

### **API Endpoints**

#### **Single Prediction**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "study_hours_per_week": 25,
    "attendance_percentage": 95,
    "sleep_hours_per_night": 8,
    "previous_grades": 85,
    "extracurricular_activities": 2,
    "parental_education_level": 2,
    "family_income": 3,
    "stress_level": 6,
    "motivation_level": 8,
    "tutoring_sessions": 1
  }'
```

#### **Batch Predictions**
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {
        "study_hours_per_week": 25,
        "attendance_percentage": 95,
        "sleep_hours_per_night": 8,
        "previous_grades": 85,
        "extracurricular_activities": 2,
        "parental_education_level": 2,
        "family_income": 3,
        "stress_level": 6,
        "motivation_level": 8,
        "tutoring_sessions": 1
      }
    ]
  }'
```

#### **Upload New Data**
```bash
curl -X POST "http://localhost:8000/upload-data" \
  -F "file=@new_students.csv"
```

#### **Check Training Status**
```bash
curl http://localhost:8000/training-status
```

#### **Health Check**
```bash
curl http://localhost:8000/health
```

### **Run Monitoring Separately**
```bash
# Run only the monitoring service
python monitor.py --interval 600  # Check every 10 minutes
```

### **Production Deployment**
```bash
# For production (no auto-reload)
python start_server.py --no-init --host 0.0.0.0 --port 80

# Or use gunicorn
pip install gunicorn
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:80
```

## 🐳 Docker Deployment

### **Build and Run with Docker**
```bash
# Build the image
docker build -t student-score-api .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts student-score-api
```

### **Run with Docker Compose**
```bash
# Start API + MLflow server
docker-compose up -d

# View logs
docker-compose logs -f student-score-api

# Stop services
docker-compose down
```

## 📱 API Client Example

Use the included Python client to interact with the API:

```bash
# Run example predictions
python api_client.py --run-example

# Create sample data
python api_client.py --create-sample

# Upload new training data (triggers retraining)
python api_client.py --upload-sample
```

## 🧪 Tests

```bash
pytest -q
```

## 🔬 MLflow UI

Runs are logged under the default local `./mlruns` unless `mlflow.tracking_uri` is set in `configs/config.yaml`.

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000`.

## ♻️ DVC pipeline (reproducible)

```bash
dvc init
dvc repro
```

To track the raw dataset:

```bash
dvc add data/raw/students.csv
git add data/raw/students.csv.dvc .gitignore
```

## 🧾 Model registry

If enabled in `configs/config.yaml`, the best model is registered to MLflow Model Registry and optionally promoted to `Staging` or `Production`.

## 🗂️ Dataset

Based on the Kaggle dataset “Student Habits vs Academic Performance”.
