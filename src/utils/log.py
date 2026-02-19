import os
import mlflow
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

def log_model_metrics(name, y_true, oof_probs):
    """Log AUC, accuracy, and per-fold metrics for a model."""
    auc = roc_auc_score(y_true, oof_probs)
    preds = (oof_probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds, output_dict=True)

    mlflow.log_metrics({
        f"{name}_auc":      auc,
        f"{name}_accuracy": acc,
        f"{name}_precision_0": report["0"]["precision"],
        f"{name}_recall_0":    report["0"]["recall"],
        f"{name}_f1_0":        report["0"]["f1-score"],
        f"{name}_precision_1": report["1"]["precision"],
        f"{name}_recall_1":    report["1"]["recall"],
        f"{name}_f1_1":        report["1"]["f1-score"],
    })

    print(f"\n  [{name.upper()}] AUC: {auc:.5f} | Accuracy: {acc:.5f}")
    print(classification_report(y_true, preds))

    return auc, acc


def save_and_log_pkl(obj, name, models_dir):
    """Save a pkl file locally and log it as an MLflow artifact."""
    path = os.path.join(models_dir, f"{name}.pkl")
    joblib.dump(obj, path)
    mlflow.log_artifact(path, artifact_path=f"models/{name}")
    print(f"  Saved & logged: {path}")
    return path
