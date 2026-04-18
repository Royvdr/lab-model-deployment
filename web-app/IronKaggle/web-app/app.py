import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("sales-model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from the HTML form
        # We need: nb_customers_on_day, promotion, school_holiday
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template(
            "index.html", 
            prediction_text=f"Predicted Sales for the Day: ${output}"
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)