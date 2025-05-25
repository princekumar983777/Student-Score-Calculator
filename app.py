from flask import Flask, render_template, request
import joblib
from src.transform_data import transform_input_data
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('student_score_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    transformed_data = transform_input_data(input_data)
    prediction = model.predict(transformed_data)
    prediction = min(prediction , 100)
    return render_template('index.html', prediction=prediction[0], form_values=input_data)


@app.route('/default/<string:mode>', methods=['GET'])
def default(mode):
    from src.default_val import default_input_data

    if mode == 'median':
        form_values = default_input_data
    elif mode == 'best':
        default_best = pd.read_csv('src/default/default_best.csv')
        random_best = default_best.sample(n=1)
        random_best_data = random_best.to_dict(orient='records')[0] 
        form_values = random_best_data
    else:
        form_values = {}
    try:
        input_data = form_values
        transformed_data = transform_input_data(input_data)
        prediction = model.predict(transformed_data)
        prediction = min(prediction , 100)
        return render_template('index.html', form_values=form_values, prediction=prediction[0])
    except Exception as e:
        # Optionally, log the error or display a message
        return render_template('index.html', form_values=form_values, prediction="Error in prediction")
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)