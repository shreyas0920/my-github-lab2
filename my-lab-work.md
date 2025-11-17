# GitHub Actions Lab 2 — Model Training & Evaluation

## Original Lab
- Trains RandomForest on synthetic data
- Evaluates F1 score
- GitHub Actions on push/schedule

## My Modifications
- **Model**: `GradientBoostingClassifier` (better for noisy data)
- **Data**: 8 features, manual Gaussian noise (`np.random.normal`)
- **Metrics**: Accuracy + Confusion Matrix + Noise Level
- **Structure**: `models/`, `metrics/`, `data/` folders
- **Notebook**: `test.ipynb` runs in `lab2_env` with full pipeline
- **Why**: More realistic, robust, production-ready

## Learnings / Challenges
- Fixed `ModuleNotFoundError: sklearn` → activated venv + `ipykernel`
- Fixed `noise` param → used `np.random.normal`

## Run Instructions
```bash
# 1. Create & activate env
python -m venv lab2_env
lab2_env\Scripts\activate
pip install -r requirements.txt

# 2. Register kernel
python -m ipykernel install --user --name lab2_env --display-name "Python (lab2_env)"

# 3. Run training
python src/train_model.py --timestamp "20251116_220600"

# 4. Run evaluation
python src/evaluate_model.py --timestamp "20251116_220600"

# 5. Open test.ipynb → Select "Python (lab2_env)" → Run All