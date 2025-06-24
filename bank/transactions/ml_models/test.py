import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# ====================
# 1. Load and Train
# ====================
def train_models(data_path: str = 'amount_only.csv') -> list[str]:
    df = pd.read_csv(data_path)

    # Define fraud by top 5% amount
    threshold = np.percentile(df['amount'], 95)
    df['label'] = (df['amount'] >= threshold).astype(int)
    print(f"Using amount >= {threshold:.2f} as fraud label (≈5% of data)")

    X = df[['amount']]
    y = df['label']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Supervised model: XGBoost (no use_label_encoder now)
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric='auc',
        verbosity=0
    )
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]

    print("--- XGBoost Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}\n")

    # Unsupervised: Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso.fit(X_train)
    anomalies = iso.predict(X_test)
    y_iso = np.where(anomalies == -1, 1, 0)

    print("--- Isolation Forest Evaluation ---")
    print(classification_report(y_test, y_iso))

    # Save models
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(xgb, 'xgb_model.pkl')
    joblib.dump(iso, 'isolation_forest.pkl')
    print("Models saved: scaler.pkl, xgb_model.pkl, isolation_forest.pkl")

    return ['amount']

# =====================================
# 2. Real-Time Prediction
# =====================================
def load_models():
    scaler = joblib.load('scaler.pkl')
    xgb = joblib.load('xgb_model.pkl')
    iso = joblib.load('isolation_forest.pkl')
    return scaler, xgb, iso

def predict_transaction(transaction: dict, scaler, xgb, iso) -> dict:
    df = pd.DataFrame([transaction])
    X_tr = scaler.transform(df)
    sup_proba = xgb.predict_proba(X_tr)[0, 1]
    iso_flag = iso.predict(X_tr)[0] == -1
    return {
        'fraud_probability': float(sup_proba),
        'anomaly_flag': iso_flag
    }

# ====================
# 3. CLI Interface
# ====================
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        features = train_models('amount_only.csv')
        print("Training done. Run without arguments for interactive mode.")
        sys.exit(0)

    features = ['amount']
    scaler, xgb, iso = load_models()

    print("Enter transaction values:")
    transaction = {}
    for feat in features:
        val = input(f"  {feat}: ")
        try:
            transaction[feat] = float(val)
        except ValueError:
            transaction[feat] = val  # support non-numeric fallback (not needed now)

    result = predict_transaction(transaction, scaler, xgb, iso)

    print("\n=== Fraud Detection Result ===")
    print(f"Fraud probability: {result['fraud_probability']:.4f}")
    print(f"Anomaly flag: {'YES' if result['anomaly_flag'] else 'NO'}")

    if result['fraud_probability'] > 0.5:
        print("⚠️  HIGH FRAUD RISK")
    else:
        print("✅  LOW RISK")
