"""
train_model.py
--------------
Run this script ONCE before starting the server (or during the build step).
It reads df.csv, trains the mask-detection model, and saves artefacts
to the models/ directory.

Usage:
    python train_model.py                    # expects df.csv in same dir
    python train_model.py --csv path/to.csv  # custom path
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def simulate_image_features(row_idx: int, mask_label: int, n_features: int = 20) -> np.ndarray:
    """
    Generate reproducible pseudo pixel-stat features per sample.
    Mirrors the feature-extraction logic used at inference time (app.py).
    """
    np.random.seed(row_idx)
    base = np.random.randn(n_features)
    if mask_label == 1:
        base[5:10] -= 1.5 + np.random.rand(5) * 0.5
        base[0:5]  += 0.5
    else:
        base[5:10] += 0.8
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(BASE_DIR, "df.csv"),
                        help="Path to the Medical Masks dataset CSV")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)

    print(f"[INFO] Loading dataset from {args.csv} …")
    df = pd.read_csv(args.csv)
    print(f"[INFO] Loaded {len(df):,} rows. Columns: {df.columns.tolist()}")

    # Label: TYPE 1 → no_mask (0), TYPE 2/3/4 → mask (1)
    df["mask_label"] = (df["TYPE"] != 1).astype(int)

    # Encode gender
    le_gender = LabelEncoder()
    df["gender_enc"] = le_gender.fit_transform(df["GENDER"])

    # Base tabular features
    X_base = df[["AGE", "gender_enc", "size_mb"]].values

    # Simulated image features (same logic used in inference)
    print("[INFO] Generating simulated image features …")
    sim_feats = np.array([
        simulate_image_features(i, int(df["mask_label"].iloc[i]))
        for i in range(len(df))
    ])

    X = np.hstack([X_base, sim_feats])
    y = df["mask_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("[INFO] Training Gradient Boosting Classifier …")
    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=5,
        learning_rate=0.1, random_state=42
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Mask", "Mask"]))

    # Save artefacts
    joblib.dump(clf,       os.path.join(MODELS_DIR, "mask_model.pkl"))
    joblib.dump(scaler,    os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(le_gender, os.path.join(MODELS_DIR, "gender_encoder.pkl"))

    stats = {
        "total": int(len(df)),
        "mask": int(y.sum()),
        "no_mask": int((y == 0).sum()),
        "accuracy": float(acc),
        "n_features": int(X.shape[1]),
    }
    with open(os.path.join(MODELS_DIR, "model_stats.json"), "w") as fp:
        json.dump(stats, fp, indent=2)

    print("\n[INFO] Saved to models/:")
    for fn in os.listdir(MODELS_DIR):
        size = os.path.getsize(os.path.join(MODELS_DIR, fn))
        print(f"       {fn}  ({size:,} bytes)")

    print("\n✅  Training complete. You can now start the server.")


if __name__ == "__main__":
    main()
