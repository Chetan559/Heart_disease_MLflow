import os
import joblib
import mlflow

from src.data.load_data import load_config, load_data
from src.preprocess.preprocess import preprocess
from src.models.train_models import train_xgb, train_lgb, train_cat, train_meta
from src.evaluate.evaluate import print_final_summary


def main():
    # ── 1. Load config ──────────────────────────────────────────────
    config = load_config("config.yaml")
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)

    # ── 2. Setup MLflow ─────────────────────────────────────────────
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # ── 3. Load & preprocess data ────────────────────────────────────
    print("Loading data...")
    train, test, test_ids = load_data(config)

    print("Preprocessing...")
    X, y, test_processed, feature_names, le = preprocess(train, test, config["target"])

    # ── 4. XGBoost run ───────────────────────────────────────────────
    print("\nTraining XGBoost...")
    with mlflow.start_run(run_name="XGBoost"):
        oof_xgb, test_xgb, xgb_model = train_xgb(X, y, test_processed, config)
        mlflow.xgboost.log_model(xgb_model, "model")

    # ── 5. LightGBM run ──────────────────────────────────────────────
    print("\nTraining LightGBM...")
    with mlflow.start_run(run_name="LightGBM"):
        oof_lgb, test_lgb, lgb_model = train_lgb(X, y, test_processed, config)
        mlflow.sklearn.log_model(lgb_model, "model")

    # ── 6. CatBoost run ──────────────────────────────────────────────
    print("\nTraining CatBoost...")
    with mlflow.start_run(run_name="CatBoost"):
        oof_cat, test_cat, cat_model = train_cat(X, y, test_processed, config)
        mlflow.sklearn.log_model(cat_model, "model")

    # ── 7. Meta Learner run ───────────────────────────────────────────
    print("\nTraining Meta Learner (Stacking)...")
    with mlflow.start_run(run_name="MetaLearner_Stack"):
        meta_model, stack_X, stack_test = train_meta(
            oof_xgb, oof_lgb, oof_cat,
            test_xgb, test_lgb, test_cat,
            y, config
        )
        mlflow.sklearn.log_model(meta_model, "model")

        feat_path = os.path.join(config["paths"]["models_dir"], "feature_names.pkl")
        joblib.dump(feature_names, feat_path)
        mlflow.log_artifact(feat_path, artifact_path="artifacts")


    # ── 8. Final summary (printed only) ──────────────────────────────
    stack_oof_pred = meta_model.predict_proba(stack_X)[:, 1]
    print_final_summary(y, oof_xgb, oof_lgb, oof_cat, stack_oof_pred)
    print("\n✅ Done! Run `mlflow ui` to compare all runs.")


if __name__ == "__main__":
    main()