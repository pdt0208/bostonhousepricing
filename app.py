import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('regmodel.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('home.html')


# API route (Postman ke liye)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']

        # Fixed feature order (VERY IMPORTANT)
        features = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
                    'DIS','RAD','TAX','PTRATIO','B','LSTAT']

        new_data = np.array([data[feature] for feature in features]).reshape(1, -1)

        prediction = model.predict(new_data)

        return jsonify({
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


# HTML form ke liye (optional but useful)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # form se values lena
        input_features = [float(x) for x in request.form.values()]
        new_data = np.array(input_features).reshape(1, -1)

        prediction = model.predict(new_data)

        return render_template('home.html', prediction_text=f"Price: {prediction[0]}")

    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")


# Run app
if __name__ == "__main__":
    app.run(debug=True)