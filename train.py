import os
import joblib
import mlflow
import dagshub
from dotenv import load_dotenv

from src.data.load_data import load_config, load_data
from src.preprocess.preprocess import preprocess
from src.models.train_cat import train_cat
from src.models.train_lgb import train_lgb
from src.models.train_xgb import train_xgb
from src.models.train_meta import train_meta


from src.evaluate.evaluate import print_final_summary


load_dotenv()
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

dagshub.init(
    repo_owner="Chetan559",
    repo_name="Heart_disease_MLflow",
    mlflow=True
)


def main():
    # Config setup
    config = load_config("config.yaml")
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)

    experiments = config["mlflow"]["experiments"]
    reg_models = config["mlflow"]["registered_models"]
    alias = config["mlflow"]["champion_alias"]
    tracking_uri = config["mlflow"]["tracking_uri"]

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    # Loading Data
    print("Loading data...")
    train, test, test_ids = load_data(config)

    print("Preprocessing...")
    X, y, test_processed, feature_names, le = preprocess(
        train, test, config["target"]
    )

    # Base Models Training
    print("\nTraining XGBoost...")
    mlflow.set_experiment(experiments["xgb"])

    with mlflow.start_run(run_name="XGBoost") as run:
        oof_xgb, test_xgb, xgb_model = train_xgb(
            X, y, test_processed, config
        )

        logged_model = mlflow.xgboost.log_model(
            xgb_model,
            name="model"
        )

        mv = mlflow.register_model(
            model_uri=logged_model.model_uri,
            name=reg_models["xgb"]
        )

        client.set_registered_model_alias(
            reg_models["xgb"],
            alias,
            mv.version
        )

        print(
            f"Registered {reg_models['xgb']} "
            f"v{mv.version} → alias='{alias}'"
        )

    print("\nTraining LightGBM...")
    mlflow.set_experiment(experiments["lgb"])

    with mlflow.start_run(run_name="LightGBM") as run:
        oof_lgb, test_lgb, lgb_model = train_lgb(
            X, y, test_processed, config
        )

        logged_model = mlflow.sklearn.log_model(
            lgb_model,
            name="model"
        )

        mv = mlflow.register_model(
            model_uri=logged_model.model_uri,
            name=reg_models["lgb"]
        )

        client.set_registered_model_alias(
            reg_models["lgb"],
            alias,
            mv.version
        )

        print(
            f"Registered {reg_models['lgb']} "
            f"v{mv.version} → alias='{alias}'"
        )

    print("\nTraining CatBoost...")
    mlflow.set_experiment(experiments["cat"])

    with mlflow.start_run(run_name="CatBoost") as run:
        oof_cat, test_cat, cat_model = train_cat(
            X, y, test_processed, config
        )

        logged_model = mlflow.sklearn.log_model(
            cat_model,
            name="model"
        )

        mv = mlflow.register_model(
            model_uri=logged_model.model_uri,
            name=reg_models["cat"]
        )

        client.set_registered_model_alias(
            reg_models["cat"],
            alias,
            mv.version
        )

        print(
            f"Registered {reg_models['cat']} "
            f"v{mv.version} → alias='{alias}'"
        )

    # Meta Learner (Stacking) Training
    print("\nTraining Meta Learner (Stacking)...")
    mlflow.set_experiment(experiments["meta"])

    with mlflow.start_run(run_name="MetaLearner_Stack") as run:
        meta_model, stack_X, stack_test = train_meta(
            oof_xgb, oof_lgb, oof_cat,
            test_xgb, test_lgb, test_cat,
            y, config
        )

        logged_model = mlflow.sklearn.log_model(
            meta_model,
            name="model"
        )

        feat_path = os.path.join(
            config["paths"]["models_dir"],
            "feature_names.pkl"
        )

        joblib.dump(feature_names, feat_path)
        mlflow.log_artifact(feat_path)

        mv = mlflow.register_model(
            model_uri=logged_model.model_uri,
            name=reg_models["meta"]
        )

        client.set_registered_model_alias(
            reg_models["meta"],
            alias,
            mv.version
        )

        print(
            f"Registered {reg_models['meta']} "
            f"v{mv.version} → alias='{alias}'"
        )

    # Final Summary 
    stack_oof_pred = meta_model.predict_proba(stack_X)[:, 1]

    print_final_summary(
        y,
        oof_xgb,
        oof_lgb,
        oof_cat,
        stack_oof_pred
    )

    print(f"\nDone! All models registered with alias='{alias}'")
    print("Open DagsHub → MLflow to compare experiments.")


if __name__ == "__main__":
    main()