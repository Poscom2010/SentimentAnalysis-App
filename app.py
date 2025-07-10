# Load the slibraries
from flask import Flask, request, jsonify
import pickle
from preprocess import preprocess_text

#Create a flask app
app = Flask (__name__)

# Load the model and vectorizer

with open ("model.pkl", "rb") as f:
    model = pickle.load(f)
with open ("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Define the prediction endpoint
@app.route('/')
def home():
    return "API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['review']
    cleaned_review = preprocess_text(data)
    X = vectorizer.transform([cleaned_review])
    prediction = model.predict(X)[0]
  
    # Map 0/1 to labels
    label_map = {0: "negative", 1: "positive"}
    sentiment_label = label_map[int(prediction)]

    return jsonify({'sentiment': sentiment_label})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)


