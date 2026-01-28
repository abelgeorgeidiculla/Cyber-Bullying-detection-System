from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained multi-label model
model = joblib.load("final_model.pkl")  # ✅ Use correct model filename

# Define category labels (should match training labels order)
categories = ['insult', 'threat', 'identity_hate', 'gender', 'religion', 'ethnicity']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form.get("text", "")
        if not text.strip():
            return render_template("index.html", prediction_text="⚠️ Please enter some text to analyze.")

        # Predict using model
        prediction = model.predict([text])[0]  # Multi-label output (e.g., [1, 0, 1, 0, 0, 1])

        # Find matching categories
        result = [categories[i] for i, val in enumerate(prediction) if val == 1]

        if result:
            output = f"🚨 Cyberbullying Detected in the categories: <strong>{', '.join(result)}</strong>"
        else:
            output = "✅ No cyberbullying detected. 😊"

        return render_template("index.html", prediction_text=output, input_text=text)

if __name__ == "__main__":
    app.run(debug=True)
