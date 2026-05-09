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

        age                   = int(data.get("age", 60))
        time_in_hospital      = int(data.get("time_in_hospital", 4))
        num_medications       = int(data.get("num_medications", 14))
        num_prior_visits      = int(data.get("num_prior_visits", 0))
        insulin_adjusted      = int(data.get("insulin_adjusted", 0))
        primary_diag_diabetes = int(data.get("primary_diag_diabetes", 0))
        a1c_tested            = int(data.get("a1c_tested", 1))
        num_lab_procedures    = int(data.get("num_lab_procedures", 45))

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

        prob = float(rf_model.predict_proba([features])[0][1])

        if prob >= HIGH_THRESHOLD:
            tier   = "High Risk"
            label  = "HIGH RISK"
            action = ("Immediate care coordinator follow-up within 24 hours. "
                      "Consider home health referral.")
        elif prob >= MED_THRESHOLD:
            tier   = "Medium Risk"
            label  = "MEDIUM RISK"
            action = ("Schedule follow-up call within 72 hours. "
                      "Ensure medication reconciliation at discharge.")
        else:
            tier   = "Low Risk"
            label  = "LOW RISK"
            action = ("Standard discharge instructions sufficient. "
                      "Routine 30-day follow-up recommended.")

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
            "risk_label":         label,
            "recommended_action": action,
            "risk_factors":       risk_factors,
            "model_info":         "Random Forest · 100 trees · AUC 0.67"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/openapi.json")
def openapi_spec():
    base_url = request.host_url.rstrip("/")
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "Readmission Risk Prediction API",
            "version": "1.0.0",
            "description": (
                "Predicts 30-day hospital readmission risk for diabetic "
                "patients using a Random Forest model trained on 101,766 "
                "patient encounters from 130 US hospitals."
            )
        },
        "servers": [{"url": base_url}],
        "paths": {
            "/predict": {
                "post": {
                    "operationId": "predictReadmissionRisk",
                    "summary": "Score a patient for 30-day readmission risk",
                    "description": (
                        "Provide patient variables and receive a real-time "
                        "readmission risk score from the trained Random Forest model."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": [
                                        "age", "time_in_hospital",
                                        "num_medications", "num_prior_visits",
                                        "insulin_adjusted",
                                        "primary_diag_diabetes", "a1c_tested"
                                    ],
                                    "properties": {
                                        "age": {
                                            "type": "integer",
                                            "description": "Patient age in years"
                                        },
                                        "time_in_hospital": {
                                            "type": "integer",
                                            "description": "Length of hospital stay in days"
                                        },
                                        "num_medications": {
                                            "type": "integer",
                                            "description": "Number of medications prescribed"
                                        },
                                        "num_prior_visits": {
                                            "type": "integer",
                                            "description": "Number of prior inpatient visits in past year"
                                        },
                                        "insulin_adjusted": {
                                            "type": "integer",
                                            "description": "Was insulin adjusted this admission? 1=Yes 0=No"
                                        },
                                        "primary_diag_diabetes": {
                                            "type": "integer",
                                            "description": "Is primary diagnosis diabetes-related? 1=Yes 0=No"
                                        },
                                        "a1c_tested": {
                                            "type": "integer",
                                            "description": "Was A1C tested this admission? 1=Yes 0=No"
                                        },
                                        "num_lab_procedures": {
                                            "type": "integer",
                                            "description": "Number of lab procedures (optional, defaults to 45)"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Risk assessment result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "risk_score": {
                                                "type": "number",
                                                "description": "Readmission probability (0-1)"
                                            },
                                            "risk_score_pct": {
                                                "type": "string",
                                                "description": "Risk score as percentage"
                                            },
                                            "risk_tier": {
                                                "type": "string",
                                                "description": "High Risk / Medium Risk / Low Risk"
                                            },
                                            "recommended_action": {
                                                "type": "string",
                                                "description": "Recommended clinical action"
                                            },
                                            "risk_factors": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "Contributing risk factors"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
