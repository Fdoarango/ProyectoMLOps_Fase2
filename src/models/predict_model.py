import joblib
import pandas as pd

def predict_anomalies(df: pd.DataFrame, model_path="models/pipeline_optimizado_mibici.pkl"):
    pipeline = joblib.load(model_path)
    predictions = pipeline.predict(df)
    scores = pipeline.decision_function(df)
    
    df["es_anomalo"] = predictions == -1
    df["score_anomalia"] = scores
    return df