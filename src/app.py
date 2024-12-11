from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (make sure the model is saved as 'rf_model.pkl')
model = joblib.load('D:\meet\predict\meet/rf_model.pkl')

# Load LabelEncoder (make sure to save and load the LabelEncoder in a similar way)
label_encoder = joblib.load('D:\meet\predict\meet\label_encoder.pkl')

@app.route('/')
def home():
    return render_template('meet/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received form data:", request.form)
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        bmi = float(request.form['bmi'])

        input_data = np.array([[age, height, weight, bmi]])
        print("Input data for prediction:", input_data)

        prediction = model.predict(input_data)
        print("Raw prediction:", prediction)

        bmi_class = label_encoder.inverse_transform(prediction)
        print("Decoded prediction:", bmi_class)

        return render_template('meet/index.html', prediction_text=f'The person is classified as: {bmi_class[0]}')

    except Exception as e:
        print("Error during prediction:", e)
        return render_template('meet/index.html', prediction_text="Error occurred, please check your inputs.")


if __name__ == "__main__":
    app.run(debug=True)

#app.run(port=5001)

