import os
import numpy as np
import pandas as pd
import streamlit as st
import mlflow
import dagshub
from src.data.load_data import load_config
from dotenv import load_dotenv


load_dotenv()
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

dagshub.init(
    repo_owner="Chetan559",
    repo_name="Heart_disease_MLflow",
    mlflow=True
)

config = load_config("config.yaml")
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

ALIAS    = config["mlflow"]["champion_alias_prod"]
BASE_ALIAS    = config["mlflow"]["champion_alias"]
reg      = config["mlflow"]["registered_models"]


def get_champion_model_key():
    """Figure out which model currently holds the champion alias."""
    client = mlflow.MlflowClient()
    for key, name in reg.items():
        try:
            client.get_model_version_by_alias(name, BASE_ALIAS)
            return key, name
        except Exception:
            continue
    raise ValueError(f"No model found with alias='{BASE_ALIAS}'")


@st.cache_resource
def load_champion():
    key, name = get_champion_model_key()
    model_uri = f"models:/{name}@{BASE_ALIAS }"

    if key == "xgb":
        model = mlflow.xgboost.load_model(model_uri)
    else:
        model = mlflow.sklearn.load_model(model_uri)

    return key, model


@st.cache_resource
def load_base_models():
    """Only loaded when champion is the meta learner."""
    xgb = mlflow.xgboost.load_model(f"models:/{reg['xgb']}@{BASE_ALIAS }")
    lgb = mlflow.sklearn.load_model(f"models:/{reg['lgb']}@{BASE_ALIAS }")
    cat = mlflow.sklearn.load_model(f"models:/{reg['cat']}@{BASE_ALIAS }")
    return xgb, lgb, cat


def predict(input_df):
    champion_key, champion_model = load_champion()

    if champion_key == "meta":
        # Stack: run all 3 base models, feed their probs into meta
        xgb, lgb, cat = load_base_models()
        stack_input = np.vstack([
            xgb.predict_proba(input_df)[:, 1],
            lgb.predict_proba(input_df)[:, 1],
            cat.predict_proba(input_df)[:, 1],
        ]).T
        prob = champion_model.predict_proba(stack_input)[0, 1]

    else:
        # Single model: xgb, lgb, or cat directly
        prob = champion_model.predict_proba(input_df)[0, 1]

    return prob, champion_key


# UI 
st.title("Heart Disease Prediction")

champion_key, _ = load_champion()
model_label = {
    "xgb":  "XGBoost",
    "lgb":  "LightGBM",
    "cat":  "CatBoost",
    "meta": "Stacked Ensemble (XGB + LGB + CAT → LR)",
}
st.markdown(f"**Active Champion:** `{model_label[champion_key]}`  |  alias: `{ALIAS}`")

st.sidebar.header("Patient Features")
age              = st.sidebar.slider("Age", 20, 90, 50)
sex              = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
chest_pain       = st.sidebar.selectbox("Chest Pain Type", [1, 2, 3, 4])
bp               = st.sidebar.slider("BP", 80, 200, 120)
cholesterol      = st.sidebar.slider("Cholesterol", 100, 600, 200)
fbs              = st.sidebar.selectbox("FBS over 120", [0, 1])
ekg              = st.sidebar.selectbox("EKG Results", [0, 1, 2])
max_hr           = st.sidebar.slider("Max HR", 60, 220, 150)
exercise_angina  = st.sidebar.selectbox("Exercise Angina", [0, 1])
st_depression    = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
slope_of_st      = st.sidebar.selectbox("Slope of ST", [1, 2, 3])
vessels_fluro    = st.sidebar.selectbox("Number of Vessels Fluro", [0, 1, 2, 3])
thallium         = st.sidebar.selectbox("Thallium", [3, 6, 7])

input_data = pd.DataFrame([{
    "Age": age, "Sex": sex, "Chest pain type": chest_pain,
    "BP": bp, "Cholesterol": cholesterol, "FBS over 120": fbs,
    "EKG results": ekg, "Max HR": max_hr, "Exercise angina": exercise_angina,
    "ST depression": st_depression, "Slope of ST": slope_of_st,
    "Number of vessels fluro": vessels_fluro, "Thallium": thallium
}])

if st.button("Predict"):
    with st.spinner("Predicting..."):
        prob, champion_key = predict(input_data)

    pred = int(prob >= 0.5)
    st.subheader("Result")
    if pred == 1:
        st.error(f"⚠️ High risk of Heart Disease  (probability: {prob:.2%})")
    else:
        st.success(f"✅ Low risk of Heart Disease  (probability: {prob:.2%})")
    st.progress(float(prob))