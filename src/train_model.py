# src/train_model.py
import os
import pickle
import random
import numpy as np
import mlflow
import datetime
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import argparse

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Training started with timestamp: {timestamp}")

    # === GENERATE TRAINING DATA ===
    X, y = make_classification(
        n_samples=random.randint(1000, 3000),
        n_features=8,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        random_state=42,
        shuffle=True
    )

    # === ADD GAUSSIAN NOISE ===
    noise_level = 0.1
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise

    # === SAVE DATA ===
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'data.pickle'), 'wb') as f:
        pickle.dump(X_noisy, f)
    with open(os.path.join(data_dir, 'target.pickle'), 'wb') as f:
        pickle.dump(y, f)

    # === MLFLOW TRACKING ===
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Synthetic_Noisy_8F"
    experiment_name = f"{dataset_name}_{timestamp}"
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=dataset_name):
        params = {
            "dataset": dataset_name,
            "samples": X_noisy.shape[0],
            "features": X_noisy.shape[1],
            "noise_level": noise_level
        }
        mlflow.log_params(params)

        # === TRAIN MODEL ===
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_noisy, y)

        y_pred = model.predict(X_noisy)
        mlflow.log_metrics({
            'Accuracy': accuracy_score(y, y_pred),
            'F1_Score': f1_score(y, y_pred)
        })

        # === SAVE MODEL TO models/ ===
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)

        model_filename = f'model_{timestamp}_gbc_model.joblib'
        model_filepath = os.path.join(models_dir, model_filename)
        dump(model, model_filepath)

        print(f"Model saved: {model_filepath}")