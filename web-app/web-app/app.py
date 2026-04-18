import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

app = Flask(__name__)

# Try to load the model when the app starts, but don't crash if it's missing!
try:
    model = pickle.load(open("./ufo-model.pkl", "rb"))
except FileNotFoundError:
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # If the model hasn't been trained yet, let the user know!
    if model is None:
        return render_template("index.html", prediction_text="Model not trained yet! Please go to /train first.")

    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )

# --- THE NEW CHALLENGE ROUTE ---
@app.route("/train")
def train_model():
    global model # Tell Python we want to update the global model variable
    
    # 1. Load and clean the data directly inside Flask!
    # (Assuming the data folder is one level up from the web-app folder)
    ufos = pd.read_csv('../data/ufos.csv')
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 
                         'Country': ufos['country'],
                         'Latitude': ufos['latitude'],
                         'Longitude': ufos['longitude']})
    ufos.dropna(inplace=True)
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

    # 2. Train the model
    Selected_features = ['Seconds','Latitude','Longitude']
    X = ufos[Selected_features]
    y = ufos['Country']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    new_model = LogisticRegression(max_iter=500)
    new_model.fit(X_train, y_train)
    
    # 3. Update the global model and save a fresh pickle file
    model = new_model
    pickle.dump(model, open('./ufo-model.pkl','wb'))
    
    return "<h1>Model successfully trained and updated!</h1> <a href='/'>Go back to the predictor</a>"

if __name__ == "__main__":
    app.run(debug=True)