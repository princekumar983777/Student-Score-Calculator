from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.models.predict import predict_from_dataframe
from src.pipelines.preprocessing_pipeline import run_preprocessing_pipeline
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.common import read_yaml
from src.utils.logger import setup_logger


# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    study_hours_per_week: float = Field(..., ge=0, le=168, description="Hours studied per week")
    attendance_percentage: float = Field(..., ge=0, le=100, description="Class attendance percentage")
    sleep_hours_per_night: float = Field(..., ge=0, le=24, description="Average sleep hours per night")
    previous_grades: float = Field(..., ge=0, le=100, description="Previous exam grades average")
    extracurricular_activities: int = Field(..., ge=0, description="Number of extracurricular activities")
    parental_education_level: int = Field(..., ge=0, le=4, description="Parental education (0=None, 1=High School, 2=Bachelor, 3=Master, 4=PhD)")
    family_income: int = Field(..., ge=0, le=4, description="Family income level (0=Low, 1=Lower-Middle, 2=Middle, 3=Upper-Middle, 4=High)")
    stress_level: int = Field(..., ge=1, le=10, description="Stress level (1-10)")
    motivation_level: int = Field(..., ge=1, le=10, description="Motivation level (1-10)")
    tutoring_sessions: int = Field(..., ge=0, description="Number of tutoring sessions per month")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    students: List[PredictionRequest]


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: float
    confidence_score: Optional[float] = None
    model_version: str
    timestamp: datetime


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    model_version: str
    timestamp: datetime


class TrainingStatus(BaseModel):
    """Training status response"""
    status: str  # "idle", "training", "completed", "failed"
    last_training: Optional[datetime]
    best_model: Optional[str]
    metrics: Optional[Dict[str, Any]]
    error_message: Optional[str]


class ServerState:
    """Global server state"""
    def __init__(self):
        self.is_training = False
        self.last_training_time = None
        self.current_model_version = "v1.0"
        self.training_error = None
        self.metrics = None
        self.data_last_modified = None

    def update_training_status(self, status: str, error: str = None):
        self.is_training = (status == "training")
        if status == "completed":
            self.last_training_time = datetime.now()
            self.training_error = None
        elif status == "failed":
            self.training_error = error


# Global state
server_state = ServerState()

# FastAPI app
app = FastAPI(
    title="Student Score Prediction API",
    description="Automated ML service for predicting student exam scores with continuous learning",
    version="1.0.0"
)

logger = setup_logger()


def check_data_changes() -> bool:
    """Check if raw data has been modified"""
    config_path = "configs/config.yaml"
    cfg = read_yaml(config_path)
    raw_data_path = Path(cfg["paths"]["raw_data"])

    if not raw_data_path.exists():
        return False

    current_modified = raw_data_path.stat().st_mtime
    last_modified = server_state.data_last_modified

    if last_modified is None:
        server_state.data_last_modified = current_modified
        return False

    if current_modified > last_modified:
        server_state.data_last_modified = current_modified
        return True

    return False


async def retrain_model():
    """Background task to retrain the model"""
    try:
        logger.info("Starting automated retraining...")
        server_state.update_training_status("training")

        # Run preprocessing
        logger.info("Running preprocessing pipeline...")
        preprocess_result = run_preprocessing_pipeline()

        # Run training
        logger.info("Running training pipeline...")
        training_result = run_training_pipeline()

        # Update metrics
        metrics_path = "artifacts/metrics.json"
        if Path(metrics_path).exists():
            with open(metrics_path, 'r') as f:
                server_state.metrics = json.load(f)

        # Update model version
        server_state.current_model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        server_state.update_training_status("completed")
        logger.info("Retraining completed successfully")

    except Exception as e:
        error_msg = f"Retraining failed: {str(e)}"
        logger.error(error_msg)
        server_state.update_training_status("failed", error_msg)


async def monitor_data_changes():
    """Background task to monitor data changes and trigger retraining"""
    while True:
        try:
            if check_data_changes():
                logger.info("Data changes detected, triggering retraining...")
                await retrain_model()
            await asyncio.sleep(300)  # Check every 5 minutes
        except Exception as e:
            logger.error(f"Error in data monitoring: {str(e)}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying


@app.on_event("startup")
async def startup_event():
    """Initialize server and start background tasks"""
    logger.info("Starting Student Score Prediction Server...")

    # Initialize data timestamp
    config_path = "configs/config.yaml"
    cfg = read_yaml(config_path)
    raw_data_path = Path(cfg["paths"]["raw_data"])
    if raw_data_path.exists():
        server_state.data_last_modified = raw_data_path.stat().st_mtime

    # Start data monitoring task
    asyncio.create_task(monitor_data_changes())

    logger.info("Server initialized and monitoring started")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Student Score Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - Single prediction",
            "batch_predict": "POST /batch-predict - Multiple predictions",
            "training_status": "GET /training-status - Check training status",
            "retrain": "POST /retrain - Manual retraining",
            "upload_data": "POST /upload-data - Upload new training data"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Make a single prediction"""
    try:
        # Convert request to DataFrame
        data = {
            "study_hours_per_week": [request.study_hours_per_week],
            "attendance_percentage": [request.attendance_percentage],
            "sleep_hours_per_night": [request.sleep_hours_per_night],
            "previous_grades": [request.previous_grades],
            "extracurricular_activities": [request.extracurricular_activities],
            "parental_education_level": [request.parental_education_level],
            "family_income": [request.family_income],
            "stress_level": [request.stress_level],
            "motivation_level": [request.motivation_level],
            "tutoring_sessions": [request.tutoring_sessions]
        }
        df = pd.DataFrame(data)

        # Make prediction
        cfg = read_yaml("configs/config.yaml")
        predictions = predict_from_dataframe(
            df=df,
            preprocessor_path=cfg["paths"]["preprocessor_path"],
            models_dir=cfg["paths"]["models_dir"],
            drop_cols=cfg["data"].get("drop_cols", [])
        )

        return PredictionResponse(
            prediction=float(predictions[0]),
            model_version=server_state.current_model_version,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    try:
        # Convert requests to DataFrame
        data = {}
        for field in PredictionRequest.__fields__:
            data[field] = [getattr(student, field) for student in request.students]

        df = pd.DataFrame(data)

        # Make predictions
        cfg = read_yaml("configs/config.yaml")
        predictions = predict_from_dataframe(
            df=df,
            preprocessor_path=cfg["paths"]["preprocessor_path"],
            models_dir=cfg["paths"]["models_dir"],
            drop_cols=cfg["data"].get("drop_cols", [])
        )

        # Create response
        prediction_responses = [
            PredictionResponse(
                prediction=float(pred),
                model_version=server_state.current_model_version,
                timestamp=datetime.now()
            )
            for pred in predictions
        ]

        return BatchPredictionResponse(
            predictions=prediction_responses,
            model_version=server_state.current_model_version,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/training-status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    return TrainingStatus(
        status="training" if server_state.is_training else "idle",
        last_training=server_state.last_training_time,
        best_model=server_state.current_model_version,
        metrics=server_state.metrics,
        error_message=server_state.training_error
    )


@app.post("/retrain")
async def manual_retrain(background_tasks: BackgroundTasks):
    """Manually trigger model retraining"""
    if server_state.is_training:
        raise HTTPException(status_code=409, detail="Training already in progress")

    background_tasks.add_task(retrain_model)
    return {"message": "Retraining started in background"}


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload new training data"""
    try:
        # Validate file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")

        # Save uploaded file
        cfg = read_yaml("configs/config.yaml")
        raw_data_path = Path(cfg["paths"]["raw_data"])
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        with open(raw_data_path, 'wb') as f:
            f.write(content)

        # Trigger retraining
        await retrain_model()

        return {"message": "Data uploaded and retraining started"}

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_version": server_state.current_model_version,
        "is_training": server_state.is_training
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )