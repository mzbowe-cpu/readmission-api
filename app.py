from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "readmission_model.pkl")
with open(MODEL_PATH, "rb") as f:
    package = pickle.load(f)

rf_model        = package["model"]
feature_columns = package["feature_columns"]
feature_medians = package["feature_medians"]
HIGH_THRESHOLD  = package["high_threshold"]
MED_THRESHOLD   = package["med_threshold"]
print(f"Model loaded — {len(feature_columns)} features")


@app.route("/")
def root():
    return jsonify({"status": "Readmission Risk API is running"})


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "features": len(feature_columns),
        "high_threshold": round(HIGH_THRESHOLD, 4),
        "med_threshold":  round(MED_THRESHOLD, 4)
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Required fields
        age                   = int(data.get("age", 60))
        time_in_hospital      = int(data.get("time_in_hospital", 4))
        num_medications       = int(data.get("num_medications", 14))
        num_prior_visits      = int(data.get("num_prior_visits", 0))
        insulin_adjusted      = int(data.get("insulin_adjusted", 0))
        primary_diag_diabetes = int(data.get("primary_diag_diabetes", 0))
        a1c_tested            = int(data.get("a1c_tested", 1))
        num_lab_procedures    = int(data.get("num_lab_procedures", 45))

        # Build feature vector from training medians
        features = pd.Series(feature_medians)
        features["age"]                   = age
        features["time_in_hospital"]      = time_in_hospital
        features["num_medications"]       = num_medications
        features["number_inpatient"]      = num_prior_visits
        features["med_changed"]           = insulin_adjusted
        features["primary_diag_diabetes"] = primary_diag_diabetes
        features["num_lab_procedures"]    = num_lab_procedures
        features["complexity_score"]      = num_lab_procedures + num_medications
        features = features.reindex(feature_columns, fill_value=0)

        # Score with Random Forest
        prob = float(rf_model.predict_proba([features])[0][1])

        # Risk tier
        if prob >= HIGH_THRESHOLD:
            tier   = "High Risk"
            emoji  = "HIGH RISK"
            action = ("Immediate care coordinator follow-up within 24 hours. "
                      "Consider home health referral.")
        elif prob >= MED_THRESHOLD:
            tier   = "Medium Risk"
            emoji  = "MEDIUM RISK"
            action = ("Schedule follow-up call within 72 hours. "
                      "Ensure medication reconciliation at discharge.")
        else:
            tier   = "Low Risk"
            emoji  = "LOW RISK"
            action = ("Standard discharge instructions sufficient. "
                      "Routine 30-day follow-up recommended.")

        # Risk factors
        risk_factors = []
        if age >= 65:
            risk_factors.append(f"Age over 65 ({age} years)")
        if time_in_hospital >= 7:
            risk_factors.append(f"Extended stay ({time_in_hospital} days)")
        if num_medications >= 15:
            risk_factors.append(f"High medication burden ({num_medications} meds)")
        if num_prior_visits >= 2:
            risk_factors.append(f"Multiple prior admissions ({num_prior_visits})")
        if insulin_adjusted:
            risk_factors.append("Insulin adjusted this admission")
        if not a1c_tested:
            risk_factors.append("A1C not tested this admission")
        if primary_diag_diabetes:
            risk_factors.append("Primary diagnosis is diabetes-related")
        if not risk_factors:
            risk_factors.append("No major risk factors identified")

        return jsonify({
            "risk_score":         round(prob, 3),
            "risk_score_pct":     f"{round(prob * 100, 1)}%",
            "risk_tier":          tier,
            "risk_label":         emoji,
            "recommended_action": action,
            "risk_factors":       risk_factors,
            "model_info":         "Random Forest · 100 trees · AUC 0.67"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
