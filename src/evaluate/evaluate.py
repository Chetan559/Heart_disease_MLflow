import mlflow
from sklearn.metrics import roc_auc_score, accuracy_score


def print_final_summary(y_true, oof_xgb, oof_lgb, oof_cat, stack_oof_pred):
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    models = {
        "XGBoost":    oof_xgb,
        "LightGBM":   oof_lgb,
        "CatBoost":   oof_cat,
        "Stack (LR)": stack_oof_pred,
    }
    best_auc = 0
    best_name = ""
    for name, probs in models.items():
        auc = roc_auc_score(y_true, probs)
        acc = accuracy_score(y_true, (probs >= 0.5).astype(int))
        print(f"  {name:<15} AUC: {auc:.5f}  Accuracy: {acc:.5f}")
        if auc > best_auc:
            best_auc = auc
            best_name = name

    print(f"\n  Best model: {best_name} (AUC={best_auc:.5f})")
    print("="*50)

    mlflow.log_metric("best_auc", best_auc)
    return best_auc