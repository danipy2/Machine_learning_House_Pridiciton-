# ============================================================
# HOUSE PRICE PREDICTION FLASK API
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os


# ============================================================
# 1. CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Allow requests from React frontend
CORS(app)


# ============================================================
# 2. LOAD THE BEST TRAINED MODEL
# ============================================================

# Get the directory where this app.py file is located
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Build the path to the saved model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_house_price_model.joblib"
)


# Load the complete trained pipeline
model = joblib.load(
    MODEL_PATH
)


print("=" * 70)
print("HOUSE PRICE PREDICTION API")
print("=" * 70)

print(
    "Best model loaded successfully!"
)

print(
    "Model path:",
    MODEL_PATH
)


# ============================================================
# 3. HOME ENDPOINT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "success": True,

        "message":
            "House Price Prediction API is running",

        "endpoint":
            "/predict"

    })


# ============================================================
# 4. PREDICTION ENDPOINT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Get JSON data sent by React
        # ----------------------------------------------------

        data = request.get_json()

        # Check if data was received
        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No input data received"

            }), 400


        # ----------------------------------------------------
        # Convert input JSON to Pandas DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame([
            data
        ])


        # ----------------------------------------------------
        # Make prediction
        #
        # The saved pipeline automatically performs:
        #
        # 1. Missing value handling
        # 2. Encoding
        # 3. Scaling
        # 4. Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # Return prediction to React
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "predicted_price":
                round(
                    float(prediction),
                    2
                )

        })


    except Exception as e:

        # ----------------------------------------------------
        # Return error if prediction fails
        # ----------------------------------------------------

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 400


# ============================================================
# 5. RUN FLASK SERVER
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )