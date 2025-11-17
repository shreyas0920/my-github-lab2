# src/evaluate_model.py
import os
import json
import random
import numpy as np
import joblib
import argparse
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp

    # === LOAD MODEL FROM models/ ===
    model_filename = f'model_{timestamp}_gbc_model.joblib'
    model_path = os.path.join('models', model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    print(f"Model loaded: {model_path}")

    # === GENERATE TEST DATA ===
    X, y = make_classification(
        n_samples=random.randint(1000, 3000),
        n_features=8,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        random_state=42,  # Fixed for reproducibility
        shuffle=True
    )

    # === ADD GAUSSIAN NOISE ===
    noise_level = 0.1
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise

    # === PREDICT & EVALUATE ===
    y_pred = model.predict(X_noisy)

    metrics = {
        "F1_Score": float(f1_score(y, y_pred)),
        "Accuracy": float(accuracy_score(y, y_pred)),
        "Confusion_Matrix": confusion_matrix(y, y_pred).tolist(),
        "Noise_Level": noise_level,
        "Test_Samples": len(y)
    }

    # === SAVE METRICS TO metrics/ ===
    metrics_dir = 'metrics'
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_filename = f'{timestamp}_metrics.json'
    metrics_filepath = os.path.join(metrics_dir, metrics_filename)

    with open(metrics_filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved: {metrics_filepath}")