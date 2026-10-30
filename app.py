from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "model/loan_approval_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        row = {
            "Gender": data["Gender"],
            "Married": data["Married"],
            "Dependents": data["Dependents"],
            "Education": data["Education"],
            "Self_Employed": data["Self_Employed"],
            "ApplicantIncome": float(data["ApplicantIncome"]),
            "CoapplicantIncome": float(data["CoapplicantIncome"]),
            "LoanAmount": float(data["LoanAmount"]),
            "Loan_Amount_Term": float(data["Loan_Amount_Term"]),
            "Credit_History": float(data["Credit_History"]),
            "Property_Area": data["Property_Area"],
        }

        X = pd.DataFrame([row], columns=FEATURES)
        prediction = model.predict(X)[0]
        probability = float(max(model.predict_proba(X)[0]))

        approved = str(prediction).lower() in {"y", "yes", "1", "true", "approved"}

        return jsonify({
            "approved": approved,
            "prediction": "Approved" if approved else "Rejected",
            "probability": round(probability * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
