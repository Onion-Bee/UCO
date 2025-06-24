# transactions/fraud.py
import pandas as pd
import joblib
from pathlib import Path

# adjust path to where your .pkl files actually live
BASE = Path(__file__).parent
ARTIFACTS = BASE / "ml_models"

scaler = joblib.load(ARTIFACTS / "scaler.pkl")
xgb    = joblib.load(ARTIFACTS / "xgb_model.pkl")
iso    = joblib.load(ARTIFACTS / "isolation_forest.pkl")

def check_fraud(amount: float, threshold: float = 0.5):
    """
    Returns: (is_fraud: bool, prob_fraud: float, is_anomaly: bool)
    """
    df = pd.DataFrame([{"amount": amount}])
    X_tr = scaler.transform(df)
    prob    = float(xgb.predict_proba(X_tr)[0, 1])
    anomaly = (iso.predict(X_tr)[0] == -1)
    is_fraud = (prob > threshold) or anomaly
    return is_fraud, prob, anomaly