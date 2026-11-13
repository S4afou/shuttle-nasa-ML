import pandas as pd
import numpy as np
import joblib
import wandb
import random
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score

# 1. INITIALIZE W&B (Experiment Tracking)
run = wandb.init(project="nasa-shuttle-mlops", job_type="train_and_register")

print("--- 1. DATA VERSIONING ---")
# Fetch Data
shuttle = fetch_openml("shuttle", version=1, parser="auto")
X = shuttle.data
y = shuttle.target
df = X.copy()
df['class'] = y

# Save Raw Data locally
df.to_csv("raw_shuttle.csv", index=False)

# Log to W&B Artifacts (Data Versioning)
artifact = wandb.Artifact(name="raw_data", type="dataset", description="Raw Shuttle Telemetry")
artifact.add_file("raw_shuttle.csv")
run.log_artifact(artifact)

# Preprocessing
if 'A1' in df.columns: df = df.drop('A1', axis=1)
# Map Class: 1 -> 0 (Normal), Others -> 1 (Anomaly)
df['label'] = df['class'].apply(lambda x: 0 if str(x) == '1' else 1)
df = df.drop('class', axis=1).drop_duplicates()

X_final = df.drop('label', axis=1)
y_final = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

# Filter for Semi-Supervised LOF
X_train_normal = X_train[y_train == 0]

print("--- 2. HYPERPARAMETER TUNING (Experiment Tracking) ---")
# We simulate the loop you did in the notebook, but log to W&B
best_f1 = 0
best_model = None
best_params = {}

# Run 5 random experiments
for i in range(5):
    # Random params
    n_neighbors = random.randint(5, 50)
    contamination = random.uniform(0.001, 0.05)
    
    # Track metrics for this specific run config
    with wandb.init(project="nasa-shuttle-mlops", group="tuning", job_type="eval") as nested_run:
        config = {
            "model": "LOF",
            "n_neighbors": n_neighbors,
            "contamination": contamination
        }
        nested_run.config.update(config)
        
        # Train Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True))
        ])
        pipeline.fit(X_train_normal)
        
        # Evaluate
        preds = pipeline.predict(X_test)
        preds = [1 if x == -1 else 0 for x in preds]
        
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        # Log to Cloud
        nested_run.log({"f1_score": f1, "precision": prec, "recall": rec})
        
        # Save best locally
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_params = config

print(f"--- 3. MODEL REGISTRY (Best F1: {best_f1:.4f}) ---")
# Save the best model
joblib.dump(best_model, "shuttle_lof_pipeline.pkl")

# Register to W&B (Model Versioning)
model_artifact = wandb.Artifact("production_lof_model", type="model")
model_artifact.add_file("shuttle_lof_pipeline.pkl")
run.log_artifact(model_artifact)

print("✅ MLOps Pipeline Complete. Check W&B Dashboard.")
wandb.finish()