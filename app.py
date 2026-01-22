# TITANIC SURVIVAL PREDICTION SYSTEM
# Flask Web GUI
# Author: Gideon Belaboh
# Matric Number: 23CG034049
# Environment: universal_env

from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load('model/titanic_survival_model.pkl')

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        try:
            # Read form input
            pclass = int(request.form["Pclass"])
            sex = 1 if request.form["Sex"].lower() == "male" else 0
            age = float(request.form["Age"])
            sibsp = int(request.form["SibSp"])
            fare = float(request.form["Fare"])

            # Arrange features in correct order
            features = np.array([[pclass, age, sibsp, fare, sex]])

            # Predict survival
            pred = model.predict(features)

            # Interpret prediction
            result = "Survived" if pred[0] == 1 else "Did Not Survive"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result)


# Single __main__ block for both local and deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = True if os.environ.get("FLASK_DEBUG", "1") == "1" else False
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
