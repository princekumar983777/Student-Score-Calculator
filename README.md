# 🎓 Student Score Predictor

A user-friendly web application that predicts a student's academic score based on various lifestyle and personal factors. Built with Python and Flask, this tool allows students to interactively explore how changes in habits—like increasing study hours or improving sleep quality—can influence their academic performance.

---

## 🌟 Features

- 📊 Predict academic performance using real-life factors
- 🔁 Test how changing habits (like more sleep or exercise) affects scores
- 🧠 Powered by a trained machine learning regression model
- 🌐 Simple and clean web interface

---

## 📥 Inputs Used for Prediction

The model considers the following student information:

- **Student Age**
- **Gender**
- **Hours of Study per Day**
- **Time Spent on Social Media**
- **Time Spent Watching Netflix/YouTube**
- **Has a Part-Time Job** (Yes/No)
- **Class Attendance Percentage**
- **Hours of Sleep per Day**
- **Diet Quality** (Poor, Average, Good)
- **Exercise Frequency** (Days per week)
- **Parental Education Level** (High School, Bachelor, Master, PhD)
- **Internet Quality** (Poor, Average, Good)
- **Mental Health Rating** (Scale 1 to 10)
- **Participation in Extracurricular Activities** (Yes/No)

---

## 📊 Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)

- **Data Cleaning**: Handled missing values and outliers.
- **Visualization**: Explored relationships between variables using plots and correlation matrices.
- **Insights**: Identified key factors influencing academic performance.

### 2️⃣ Feature Engineering

- **Encoding**: Converted categorical variables using techniques like `OrdinalEncoder`.
- **Scaling**: Normalized numerical features to improve model performance.

### 3️⃣ Model Training

- **Algorithm Selection**: Evaluated multiple regression models.
- **Final Model**: Selected a **Linear Regression** model based on performance metrics.
- **Model Saving**: Serialized the trained model using `joblib` as `model.pkl`.

### 4️⃣ Deployment

- **Web Interface**: Developed a Flask-based web application for user interaction.
- **Local Hosting**: Configured the app to run locally with the option for future cloud deployment.

---

## 📂 Dataset Reference

The model was trained using the [Student Habits vs Academic Performance](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance) dataset available on Kaggle. This dataset includes various factors such as study habits, lifestyle choices, and mental health indicators that influence academic performance.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
    git clone https://github.com/your-username/student-score-predictor.git
    cd student-score-predictor

### 2️⃣ (Optional) Create and Activate a Virtual Environment
    pip install -r requirements.txt
### 4️⃣ Run the Application
    python app.py
### 5️⃣ Open in Browser
    visit :
    http://127.0.0.1:5000
