from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = "random_forest_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load the TF-IDF Vectorizer
vectorizer_path = "tfidf_vectorizer.pkl"
with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

# Define Home Page Route
@app.route("/")
def home():
    return render_template("index.html")

# Define Predict Route
@app.route("/predict", methods=["POST"])
def predict():
    # Get input sentence from form
    sentence = request.form["sentence"]

    # Transform input sentence using the TF-IDF vectorizer
    X_input = vectorizer.transform([sentence])  # Now matches model input

    # Predict sentiment
    prediction = model.predict(X_input)[0]  # Ensure input shape matches

    # Sentiment mapping
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    result = sentiment_map.get(prediction, "Unknown")

    return f"Predicted Sentiment: {result}"

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
