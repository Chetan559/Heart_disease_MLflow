import numpy as np
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from src.utils.log import log_model_metrics, save_and_log_pkl


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