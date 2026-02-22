# Heart Disease Prediction with MLflow

An MLOps project that predicts the probability of heart disease using an ensemble of gradient boosting models (XGBoost, LightGBM, CatBoost) and a Logistic Regression meta-learner. The project uses **MLflow** via **DagsHub** for experiment tracking and model registry, and features a **Streamlit** web application for real-time predictions.

---

## 🔗 Quick Links

- 📊 **[MLflow Dashboard](https://dagshub.com/Chetan559/Heart_disease_MLflow.mlflow/)**: View experiment tracking, model registry, and metrics.
- 🌐 **[Streamlit Web App](https://heart-disease-prediction-chetan.streamlit.app/)**: Try out the live heart disease prediction model.

---

## Features

- **Ensemble Modeling:** Trains XGBoost, LightGBM, and CatBoost models, combining them using a stacking meta-learner.
- **Experiment Tracking:** Logs metrics (AUC, Accuracy, F1), parameters, and artifacts to MLflow.
- **Model Registry:** Manages model versions using `candidate` and `champion` aliases for seamless deployment.
- **Interactive UI:** A Streamlit app that dynamically loads the active `champion` model to serve predictions.

---

## Task Done

### 1. Models Trained On

- Random Forest Classifier
- XGBoost Classifier
- CatBoost Classifier
- LightGBM Classifier

### 2. Cross Validation

- Stratified k-fold

### 3. Ensemble Learning

- random forest for bagging
- XGBoost and CatBoost for boosting
- stacking using logistic regression / ridge regression meta-learner

### 4. MLflow

- MLflow logging
- Metrics logging
- Confusion matrix
- ROC curve
- Classification report

## Project Structure

```test
Heart_disease_MLflow/
│
├── README.md                 # Project documentation
├── app.py                    # Streamlit web application for predictions
├── config.yaml               # Centralized configuration (paths, hyperparameters, MLflow)
├── requirements.txt          # Python dependencies
├── train.py                  # Main pipeline script to train all models
├── .env.example              # Example environment variables (DagsHub credentials)
│
├── scripts/                  # CLI utilities
│   └── register_models.py    # Utility to promote models using MLflow aliases
│
└── src/                      # Source code modules
    ├── evaluate/
    │   └── evaluate.py       # Evaluation metrics and final summaries
    ├── models/               # Model training scripts
    │   ├── train_cat.py      # CatBoost training script
    │   ├── train_lgb.py      # LightGBM training script
    │   ├── train_meta.py     # Logistic Regression stacking meta-learner
    │   └── train_xgb.py      # XGBoost training script
    ├── preprocess/
    │   └── preprocess.py     # Data cleaning and label encoding
    └── utils/
        └── log.py            # MLflow logging and model artifact saving utilities
```
