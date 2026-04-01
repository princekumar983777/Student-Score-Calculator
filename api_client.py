#!/usr/bin/env python3
"""
Example script showing how to interact with the Student Score Prediction API.
"""

import requests
import json
import pandas as pd
from typing import List, Dict, Any


class StudentScoreAPI:
    """Client for the Student Score Prediction API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def predict_single(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single prediction"""
        url = f"{self.base_url}/predict"
        response = requests.post(url, json=student_data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    def predict_batch(self, students_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions"""
        url = f"{self.base_url}/batch-predict"
        payload = {"students": students_data}
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    def get_training_status(self) -> Dict[str, Any]:
        """Get training status"""
        url = f"{self.base_url}/training-status"
        response = requests.get(url)
        return response.json()

    def trigger_retraining(self) -> Dict[str, Any]:
        """Manually trigger retraining"""
        url = f"{self.base_url}/retrain"
        response = requests.post(url)
        return response.json()

    def upload_data(self, csv_file_path: str) -> Dict[str, Any]:
        """Upload new training data"""
        url = f"{self.base_url}/upload-data"
        with open(csv_file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        return response.json()


def example_usage():
    """Example usage of the API client"""
    api = StudentScoreAPI()

    print("🔍 Checking server health...")
    try:
        health = api.health_check()
        print(f"✅ Server is healthy: {health}")
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        return

    print("\n📊 Making single prediction...")
    student = {
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

    try:
        result = api.predict_single(student)
        print(f"🎯 Prediction: {result['prediction']:.1f}")
        print(f"📝 Model version: {result['model_version']}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

    print("\n📊 Making batch predictions...")
    students = [
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
        },
        {
            "study_hours_per_week": 15,
            "attendance_percentage": 80,
            "sleep_hours_per_night": 6,
            "previous_grades": 70,
            "extracurricular_activities": 1,
            "parental_education_level": 1,
            "family_income": 2,
            "stress_level": 8,
            "motivation_level": 5,
            "tutoring_sessions": 0
        }
    ]

    try:
        batch_result = api.predict_batch(students)
        print(f"📈 Batch predictions completed: {len(batch_result['predictions'])} students")
        for i, pred in enumerate(batch_result['predictions']):
            print(f"  Student {i+1}: {pred['prediction']:.1f}")
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")

    print("\n📋 Checking training status...")
    try:
        status = api.get_training_status()
        print(f"🔄 Training status: {status['status']}")
        if status.get('last_training'):
            print(f"🕒 Last training: {status['last_training']}")
        if status.get('best_model'):
            print(f"🏆 Best model: {status['best_model']}")
    except Exception as e:
        print(f"❌ Status check failed: {e}")


def create_sample_data():
    """Create sample student data for testing"""
    sample_students = [
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
            "tutoring_sessions": 1,
            "exam_score": 88  # Target for training
        },
        {
            "study_hours_per_week": 15,
            "attendance_percentage": 80,
            "sleep_hours_per_night": 6,
            "previous_grades": 70,
            "extracurricular_activities": 1,
            "parental_education_level": 1,
            "family_income": 2,
            "stress_level": 8,
            "motivation_level": 5,
            "tutoring_sessions": 0,
            "exam_score": 72
        },
        {
            "study_hours_per_week": 35,
            "attendance_percentage": 98,
            "sleep_hours_per_night": 9,
            "previous_grades": 92,
            "extracurricular_activities": 3,
            "parental_education_level": 3,
            "family_income": 4,
            "stress_level": 4,
            "motivation_level": 9,
            "tutoring_sessions": 2,
            "exam_score": 95
        }
    ]

    df = pd.DataFrame(sample_students)
    df.to_csv("sample_new_students.csv", index=False)
    print("📄 Created sample_new_students.csv for testing")
    return "sample_new_students.csv"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Student Score API Client Example")
    parser.add_argument("--run-example", action="store_true",
                       help="Run the example usage")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample data file")
    parser.add_argument("--upload-sample", action="store_true",
                       help="Upload sample data to server")

    args = parser.parse_args()

    if args.create_sample:
        sample_file = create_sample_data()
        print(f"Created sample data: {sample_file}")

    elif args.upload_sample:
        api = StudentScoreAPI()
        sample_file = create_sample_data()
        try:
            result = api.upload_data(sample_file)
            print(f"✅ Upload result: {result}")
        except Exception as e:
            print(f"❌ Upload failed: {e}")

    elif args.run_example:
        example_usage()

    else:
        print("Student Score Prediction API Client")
        print("Usage:")
        print("  python api_client.py --run-example    # Run example predictions")
        print("  python api_client.py --create-sample  # Create sample data")
        print("  python api_client.py --upload-sample  # Upload sample data")