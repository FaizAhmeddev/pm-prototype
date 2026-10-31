"""Trains the three models on simulated bearing data. Run: python -m ml.train"""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml.core import FAULTS, HORIZON, features, process_tags, simulate_window

OUT = Path(__file__).resolve().parent.parent / "models"


def train(n=300, seed=0, save=True):
    rng = np.random.default_rng(seed)
    X, y, S = [], [], []
    for fault in FAULTS:
        for _ in range(n):
            sev = 0.0 if fault == "normal" else rng.uniform(0.1, 1.0)
            t, o = process_tags(sev, rng)
            X.append(features(simulate_window(fault, sev, rng=rng), t, o))
            y.append(fault)
            S.append(sev)
    X, y, R = np.array(X), np.array(y), (1 - np.array(S)) * HORIZON
    Xtr, Xte, ytr, yte, Rtr, Rte = train_test_split(X, y, R, test_size=0.25, random_state=seed, stratify=y)
    clf = RandomForestClassifier(200, random_state=seed).fit(Xtr, ytr)
    reg = GradientBoostingRegressor(random_state=seed).fit(Xtr, Rtr)
    sigma = float(np.std(Rte - reg.predict(Xte)))
    sc = StandardScaler().fit(Xtr[ytr == "normal"])
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(sc.transform(Xtr[ytr == "normal"]))
    s = -iso.score_samples(sc.transform(Xtr))
    lo, hi = float(np.percentile(s[ytr == "normal"], 95)), float(np.median(s[ytr != "normal"]))
    hi = max(hi, lo + 1e-3)
    metrics = dict(data="synthetic bearing simulation", fault_accuracy=round(float(clf.score(Xte, yte)), 3),
                   anomaly_auc=round(float(roc_auc_score(yte != "normal", -iso.score_samples(sc.transform(Xte)))), 3),
                   rul_mae_days=round(float(np.mean(np.abs(Rte - reg.predict(Xte)))), 2), rul_sigma_days=round(sigma, 2))
    bundle = dict(scaler=sc, iso=iso, clf=clf, rul=reg, sigma=sigma, lo=lo, hi=hi)
    if save:
        OUT.mkdir(exist_ok=True)
        joblib.dump(bundle, OUT / "bundle.joblib")
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)
    return bundle


if __name__ == "__main__":
    train()
