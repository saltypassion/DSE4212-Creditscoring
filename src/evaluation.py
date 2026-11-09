"""Shared evaluation utilities (DSE4212/QF4212 Group Project 1).

Convention everywhere: positive class (1) = default / bad credit.
Every model (LR, RF, AdaBoost) should be scored with `evaluate()` so all
metrics and saved probability files are identical in format.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score, confusion_matrix,
                             precision_recall_curve, classification_report)

RANDOM_STATE = 4212


def evaluate(model, X, y, threshold=0.5, name="model", fn_cost=5,
             save_to=None, report=False):
    """Score a fitted model (anything with predict_proba) on (X, y).

    Returns a dict with AUC, precision, recall, F1, accuracy, confusion
    matrix, misclassification cost and the predicted probabilities.
    If `save_to` is given, probabilities are written to that CSV.
    """
    y = np.asarray(y)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    res = {
        "name": name,
        "threshold": float(threshold),
        "auc": roc_auc_score(y, y_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "accuracy": accuracy_score(y, y_pred),
        "confusion_matrix": cm,
        "cost": float(fn_cost * fn + fp),
        "y_proba": y_proba,
    }

    print(f"=== {name} | threshold = {threshold:.3f} ===")
    print(f"AUC {res['auc']:.4f} | Precision {res['precision']:.4f} | "
          f"Recall {res['recall']:.4f} | F1 {res['f1']:.4f} | "
          f"Accuracy {res['accuracy']:.4f}")
    print(pd.DataFrame(cm, index=["actual 0", "actual 1"],
                       columns=["pred 0", "pred 1"]))
    if report:
        print(classification_report(y, y_pred, zero_division=0))

    if save_to:
        index = X.index if hasattr(X, "index") else np.arange(len(y))
        pd.DataFrame({"index": index, "y_true": y, "y_proba": y_proba}
                     ).to_csv(save_to, index=False)
    return res


def metrics_only(res):
    """JSON-friendly copy of an evaluate() result (no arrays)."""
    tn, fp, fn, tp = res["confusion_matrix"].ravel()
    keep = ["threshold", "auc", "precision", "recall", "f1", "accuracy", "cost"]
    out = {k: float(res[k]) for k in keep}
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def results_table(results, fn_cost=5):
    """One row per evaluate() result, ready to display or save as CSV."""
    rows = []
    for r in results:
        tn, fp, fn, tp = r["confusion_matrix"].ravel()
        rows.append({
            "Model / operating point": r["name"],
            "Threshold": round(r["threshold"], 3),
            "AUC": round(r["auc"], 3),
            "Precision": round(r["precision"], 3),
            "Recall": round(r["recall"], 3),
            "F1": round(r["f1"], 3),
            "Accuracy": round(r["accuracy"], 3),
            "TN/FP/FN/TP": f"{tn}/{fp}/{fn}/{tp}",
            f"Cost (FN:FP = {fn_cost}:1)": float(fn_cost * fn + fp),
        })
    return pd.DataFrame(rows)


def f1_optimal_threshold(y_true, y_proba):
    """Threshold that maximises F1 (use out-of-fold training predictions)."""
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.argmax(f1)])


def cost_min_threshold(y_true, y_proba, cost_fn=5, cost_fp=1):
    """Empirical threshold minimising cost_fn*FN + cost_fp*FP."""
    y_true = np.asarray(y_true)
    cands = np.unique(y_proba)
    costs = [cost_fn * np.sum((y_proba < t) & (y_true == 1))
             + cost_fp * np.sum((y_proba >= t) & (y_true == 0)) for t in cands]
    return float(cands[int(np.argmin(costs))])


def cost_optimal_threshold(cost_fn=5, cost_fp=1):
    """Theoretical threshold for perfectly calibrated probabilities."""
    return cost_fp / (cost_fp + cost_fn)


def bootstrap_auc_ci(y_true, y_proba, n_boot=2000, alpha=0.05,
                     random_state=RANDOM_STATE):
    """Percentile bootstrap confidence interval for the AUC."""
    rng = np.random.default_rng(random_state)
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    aucs = []
    while len(aucs) < n_boot:
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue  # resample lacks one class
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]
