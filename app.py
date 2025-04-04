from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Only 12 features now (removed DayOfYear)
        data = [float(request.form[key]) for key in [
            'LATITUDE', 'LONGITUDE', 'ELEVATION', 'AWND', 'PRCP', 'SNOW',
            'SNWD', 'TMAX', 'TMIN', 'Year', 'Month', 'Day'
        ]]

        input_data = np.array([data])
        predicted_temp = model.predict(input_data)[0]

        # Add emoji based on predicted temperature
        if predicted_temp >= 26:
            emoji = "☀️ Sunny"
        elif predicted_temp <= 15:
            emoji = "❄️ Cold"
        else:
            emoji = "🌧️ Rainy"


        return jsonify({'Predicted Temperature': f"{predicted_temp:.2f}°C {emoji}"})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
