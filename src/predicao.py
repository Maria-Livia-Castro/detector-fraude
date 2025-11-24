import joblib
import pandas as pd

def prever(transacoes):
    pipe = joblib.load("fraud_model.joblib")
    scores = pipe.predict_proba(transacoes)[:, 1]
    preds = (scores >= 0.5).astype(int)
    return pd.DataFrame({"score": scores, "is_fraud_pred": preds})