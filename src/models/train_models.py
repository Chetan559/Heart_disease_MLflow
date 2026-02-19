import numpy as np
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

from src.utils.log import log_model_metrics, save_and_log_pkl


def train_xgb(X, y, test, config):
    kf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["random_state"])
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    params = config["xgb"]
    models_dir = config["paths"]["models_dir"]
    best_auc, best_model = 0, None

    mlflow.log_params({f"xgb_{k}": v for k, v in params.items()})

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params, random_state=config["random_state"], n_jobs=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_probs = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_probs)
        mlflow.log_metric(f"xgb_fold{fold+1}_auc", fold_auc)

        oof[val_idx] = val_probs
        test_preds += model.predict_proba(test)[:, 1] / kf.n_splits

        if fold_auc > best_auc:
            best_auc, best_model = fold_auc, model

        print(f"  XGB Fold {fold+1} | AUC: {fold_auc:.5f}" + (" ✓ best" if fold_auc == best_auc else ""))

    log_model_metrics("xgb", y, oof)
    save_and_log_pkl(best_model, "xgb_best", models_dir)
    print(f"  XGB best fold AUC: {best_auc:.5f}")

    return oof, test_preds, best_model


def train_lgb(X, y, test, config):
    kf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["random_state"])
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    params = config["lgb"]
    models_dir = config["paths"]["models_dir"]
    best_auc, best_model = 0, None

    mlflow.log_params({f"lgb_{k}": v for k, v in params.items()})

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params, verbose=-1, random_state=config["random_state"], n_jobs=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(100), log_evaluation(0)]
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_probs)
        mlflow.log_metric(f"lgb_fold{fold+1}_auc", fold_auc)

        oof[val_idx] = val_probs
        test_preds += model.predict_proba(test)[:, 1] / kf.n_splits

        if fold_auc > best_auc:
            best_auc, best_model = fold_auc, model

        print(f"  LGB Fold {fold+1} | AUC: {fold_auc:.5f}" + (" ✓ best" if fold_auc == best_auc else ""))

    log_model_metrics("lgb", y, oof)
    save_and_log_pkl(best_model, "lgb_best", models_dir)
    print(f"  LGB best fold AUC: {best_auc:.5f}")

    return oof, test_preds, best_model


def train_cat(X, y, test, config):
    kf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["random_state"])
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    params = config["cat"]
    models_dir = config["paths"]["models_dir"]
    best_auc, best_model = 0, None

    mlflow.log_params({f"cat_{k}": v for k, v in params.items()})

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params, random_state=config["random_state"])
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200, verbose=False)

        val_probs = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_probs)
        mlflow.log_metric(f"cat_fold{fold+1}_auc", fold_auc)

        oof[val_idx] = val_probs
        test_preds += model.predict_proba(test)[:, 1] / kf.n_splits

        if fold_auc > best_auc:
            best_auc, best_model = fold_auc, model

        print(f"  CAT Fold {fold+1} | AUC: {fold_auc:.5f}" + (" ✓ best" if fold_auc == best_auc else ""))

    log_model_metrics("cat", y, oof)
    save_and_log_pkl(best_model, "cat_best", models_dir)
    print(f"  CAT best fold AUC: {best_auc:.5f}")

    return oof, test_preds, best_model


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