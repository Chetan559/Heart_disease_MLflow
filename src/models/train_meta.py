import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from src.utils.log import log_model_metrics, save_and_log_pkl


def train_meta(oof_xgb, oof_lgb, oof_cat, test_xgb, test_lgb, test_cat, y, config):
    """Stack base models with Logistic Regression meta-learner."""
    stack_X = np.vstack([oof_xgb, oof_lgb, oof_cat]).T
    stack_test = np.vstack([test_xgb, test_lgb, test_cat]).T

    meta_params = config["meta"]
    models_dir = config["paths"]["models_dir"]

    meta = LogisticRegression(**meta_params, random_state=config["random_state"])
    meta.fit(stack_X, y)

    mlflow.log_params({f"meta_{k}": v for k, v in meta_params.items()})

    # Log meta model coefficients
    mlflow.log_metrics({
        "meta_coef_xgb": float(meta.coef_[0][0]),
        "meta_coef_lgb": float(meta.coef_[0][1]),
        "meta_coef_cat": float(meta.coef_[0][2]),
    })

    # Log full metrics for stacked model
    stack_oof_pred = meta.predict_proba(stack_X)[:, 1]
    log_model_metrics("stack", y, stack_oof_pred)

    # Save meta model
    save_and_log_pkl(meta, "meta_learner", models_dir)

    # Save the full bundle
    bundle = {
        "xgb": oof_xgb,   # keeping oof for reference
        "lgb": oof_lgb,
        "cat": oof_cat,
        "meta": meta,
    }
    save_and_log_pkl(bundle, "full_bundle", models_dir)

    return meta, stack_X, stack_test