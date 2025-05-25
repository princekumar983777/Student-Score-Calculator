default_input_data = {
    "age": 20.0,
    "gender": "Male",
    "study_hours_per_day": 3.5,
    "social_media_hours": 2.5,
    "netflix_hours": 1.8,
    "part_time_job": "Yes",
    "attendance_percentage": 84.5,
    "sleep_hours": 6.4,
    "diet_quality": "Good",
    "exercise_frequency": 3.0,
    "parental_education_level": "Bachelor",
    "internet_quality": "Good",
    "mental_health_rating": 5.0,
    "extracurricular_participation": "No"
}
import pandas as pd
def best_input_data():
    default_best = pd.read_csv('default/default_best.csv')
    random_best = default_best.sample(n=1)
    random_best_data = random_best.to_dict(orient='records')[0] 
    return random_best_data