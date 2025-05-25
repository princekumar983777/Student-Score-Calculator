from .ordinal_columns import ordinal_columns
from .default_val import default_input_data

from sklearn.preprocessing import OrdinalEncoder
import pandas as pd 

# Input data is expected to be like this:
# {
#   "age": 22,
#   "gender": "Female",
#   "study_hours_per_day": 3.5,
#   "social_media_hours": 1.2,
#   "netflix_hours": 2.0,
#   "part_time_job": "Yes",
#   "attendance_percentage": 95.3,
#   "sleep_hours": 7.0,
#   "diet_quality": "Good",
#   "exercise_frequency": 4,
#   "parental_education_level": "Bachelor",
#   "internet_quality": "Good",
#   "mental_health_rating": 8,
#   "extracurricular_participation": "No"
# }


def transform_input_data(input_data):
    test_data = pd.DataFrame([input_data])
    # Ensure all columns are present in the DataFrame
    for col in ordinal_columns.keys():
        if col not in test_data.columns or test_data[col].isnull().any():
            test_data[col] = default_input_data[col]  # Default value if column is missing
    # Loop through and encode
    for col, order in ordinal_columns.items():
        encoder = OrdinalEncoder(categories=[order])
        test_data[[col]] = encoder.fit_transform(test_data[[col]])  # 2D input for OrdinalEncoder
    
    # Ensure all columns are present in the DataFrame
    for col in ordinal_columns.keys():
        if col not in test_data.columns:
            test_data[col] = 0  # Default value if column is missing

    return test_data   