from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
import os

# ── Load model on startup ──────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "readmission_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        package = pickle.load(f)
    rf_model        = package["model"]
    feature_columns = package["feature_columns"]
    feature_medians = package["feature_medians"]
    HIGH_THRESHOLD  = package["high_threshold"]
    MED_THRESHOLD   = package["med_threshold"]
    print(f"✅ Model loaded — {len(feature_columns)} features")
    print(f"   High Risk threshold : {HIGH_THRESHOLD:.4f}")
    print(f"   Medium Risk threshold: {MED_THRESHOLD:.4f}")
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Readmission Risk Prediction API",
    description=(
        "Predicts 30-day hospital readmission risk for diabetic patients "
        "using a Random Forest model trained on the Diabetes 130-US Hospitals "
        "dataset (101,766 patient encounters from 130 US hospitals). "
        "Returns a risk probability, tier (High/Medium/Low), and recommended "
        "clinical action."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Input schema ───────────────────────────────────────────────
class PatientInput(BaseModel):
    age: int = Field(
        ..., ge=0, le=100,
        description="Patient age in years",
        example=72
    )
    time_in_hospital: int = Field(
        ..., ge=1, le=30,
        description="Length of hospital stay in days",
        example=8
    )
    num_medications: int = Field(
        ..., ge=1, le=80,
        description="Number of medications prescribed during admission",
        example=18
    )
    num_prior_visits: int = Field(
        ..., ge=0, le=20,
        description="Number of prior inpatient visits in the past year",
        example=3
    )
    insulin_adjusted: int = Field(
        ..., ge=0, le=1,
        description="Was insulin regimen adjusted during this admission? 1=Yes 0=No",
        example=1
    )
    primary_diag_diabetes: int = Field(
        ..., ge=0, le=1,
        description="Was the primary diagnosis diabetes-related? 1=Yes 0=No",
        example=1
    )
    a1c_tested: int = Field(
        ..., ge=0, le=1,
        description="Was A1C tested during this admission? 1=Yes 0=No",
        example=0
    )
    num_lab_procedures: int = Field(
        default=45, ge=1, le=132,
        description="Number of lab procedures performed (optional, defaults to dataset median)",
        example=62
    )

# ── Output schema ──────────────────────────────────────────────
class PredictionOutput(BaseModel):
    risk_score:          float
    risk_score_pct:      str
    risk_tier:           str
    risk_emoji:          str
    recommended_action:  str
    risk_factors:        list[str]
    model_info:          str

# ── Predict endpoint ───────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Score a single patient for 30-day readmission risk",
    description=(
        "Provide key patient variables and receive a real-time readmission "
        "risk assessment from the trained Random Forest model."
    ),
)
def predict(patient: PatientInput):
    try:
        # Build feature vector from training medians
        features = pd.Series(feature_medians)

        # Override with provided patient values
        features["age"]                  = patient.age
        features["time_in_hospital"]     = patient.time_in_hospital
        features["num_medications"]      = patient.num_medications
        features["number_inpatient"]     = patient.num_prior_visits
        features["med_changed"]          = patient.insulin_adjusted
        features["primary_diag_diabetes"]= patient.primary_diag_diabetes
        features["num_lab_procedures"]   = patient.num_lab_procedures
        features["complexity_score"]     = (
            patient.num_lab_procedures +
            patient.num_medications
        )

        # Align to training feature order
        features = features.reindex(feature_columns, fill_value=0)

        # Score with Random Forest
        prob = float(rf_model.predict_proba([features])[0][1])

        # Risk tier using calibrated thresholds
        if prob >= HIGH_THRESHOLD:
            tier   = "High Risk"
            emoji  = "🔴"
            action = ("Immediate care coordinator follow-up within 24 hours. "
                      "Consider home health referral given clinical complexity.")
        elif prob >= MED_THRESHOLD:
            tier   = "Medium Risk"
            emoji  = "🟡"
            action = ("Schedule follow-up call within 72 hours. "
                      "Ensure medication reconciliation at discharge.")
        else:
            tier   = "Low Risk"
            emoji  = "🟢"
            action = ("Standard discharge instructions are sufficient. "
                      "Routine 30-day follow-up appointment recommended.")

        # Identify contributing risk factors
        risk_factors = []
        if patient.age >= 65:
            risk_factors.append(f"Age over 65 ({patient.age} years)")
        if patient.time_in_hospital >= 7:
            risk_factors.append(f"Extended hospital stay ({patient.time_in_hospital} days)")
        if patient.num_medications >= 15:
            risk_factors.append(f"High medication burden ({patient.num_medications} medications)")
        if patient.num_prior_visits >= 2:
            risk_factors.append(f"Multiple prior admissions ({patient.num_prior_visits} visits)")
        if patient.insulin_adjusted:
            risk_factors.append("Insulin regimen adjusted this admission")
        if not patient.a1c_tested:
            risk_factors.append("A1C not tested during this admission")
        if patient.primary_diag_diabetes:
            risk_factors.append("Primary diagnosis is diabetes-related")
        if not risk_factors:
            risk_factors.append("No major risk factors identified")

        return PredictionOutput(
            risk_score         = round(prob, 3),
            risk_score_pct     = f"{round(prob * 100, 1)}%",
            risk_tier          = tier,
            risk_emoji         = emoji,
            recommended_action = action,
            risk_factors       = risk_factors,
            model_info         = (
                "Random Forest · 100 trees · AUC 0.67 · "
                "Trained on Diabetes 130-US Hospitals dataset (UCI ML Repository)"
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ───────────────────────────────────────────────
@app.get(
    "/health",
    summary="Health check",
    description="Returns healthy status when the model is loaded and ready.",
)
def health():
    return {
        "status":   "healthy",
        "model":    "Random Forest Readmission Classifier",
        "features": len(feature_columns),
        "thresholds": {
            "high_risk":   round(HIGH_THRESHOLD, 4),
            "medium_risk": round(MED_THRESHOLD, 4),
        }
    }


# ── Root ───────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Readmission Risk Prediction API",
        "docs":    "/docs",
        "predict": "/predict",
        "health":  "/health",
    }
