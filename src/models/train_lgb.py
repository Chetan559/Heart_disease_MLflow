import numpy as np
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from src.utils.log import log_model_metrics, save_and_log_pkl


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