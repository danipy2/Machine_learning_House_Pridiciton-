
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os



app = Flask(__name__)

CORS(app)



BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_house_price_model.joblib"
)


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



@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "success": True,

        "message":
            "House Price Prediction API is running",

        "endpoint":
            "/predict"

    })



@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:


        data = request.get_json()

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No input data received"

            }), 400



        input_data = pd.DataFrame([
            data
        ])



        prediction = model.predict(
            input_data
        )[0]



        return jsonify({

            "success": True,

            "predicted_price":
                round(
                    float(prediction),
                    2
                )

        })


    except Exception as e:


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 400



if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )