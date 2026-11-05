"""
Lab 06 - train the model the API serves.

Builds the four-feature Titanic survival model (PClass, Gender, Sibling, Embarked)
from the Lab 05 Titanic data and saves two sets of artefacts into artifacts/:

  1. Five separate pickles: svc_trained_model.pkl + four LabelEncoder pickles
  2. One bundled Pipeline:  titanic_pipeline.joblib   (Exercise 1)

Both use the same SVC settings and the same integer codes, so they must agree.
Run from the lab06 folder:  python train_model.py
"""

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "lab05" / "data" / "titanic.csv"
OUT = HERE / "artifacts"

FEATURES = ["PClass", "Gender", "Sibling", "Embarked"]
SIBLING_WORDS = {0: "Zero", 1: "One", 2: "Two", 3: "Three"}   # 3 means "three or more"
SVC_SETTINGS = dict(probability=True, random_state=42)        # probability=True -> real confidence


def load_frame():
    """Read the Titanic table and turn it into the four categorical API features."""
    raw = pd.read_csv(DATA).dropna(subset=["embark_town"])
    return pd.DataFrame({
        "PClass": raw["pclass"].map({1: "First", 2: "Second", 3: "Third"}),
        "Gender": raw["sex"].str.capitalize(),
        "Sibling": raw["sibsp"].clip(upper=3).map(SIBLING_WORDS),
        "Embarked": raw["embark_town"],
        "Survived": raw["survived"],
    })


def main():
    OUT.mkdir(exist_ok=True)
    frame = load_frame()
    print("Rows used:", len(frame), "| survival rate:", round(frame["Survived"].mean(), 3))

    # --- approach 1: separate LabelEncoders + a bare SVC ------------------------------
    encoders = {name: LabelEncoder().fit(frame[name]) for name in FEATURES}
    codes = pd.DataFrame({name: encoders[name].transform(frame[name]) for name in FEATURES})
    y = frame["Survived"].to_numpy()

    idx_train, idx_test = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=42, stratify=y)
    svc = SVC(**SVC_SETTINGS).fit(codes.iloc[idx_train], y[idx_train])
    accuracy = accuracy_score(y[idx_test], svc.predict(codes.iloc[idx_test]))
    baseline = max(y[idx_test].mean(), 1 - y[idx_test].mean())
    print(f"Test accuracy {accuracy:.3f} vs always-majority baseline {baseline:.3f}")

    file_names = {"PClass": "pclass_encoder.pkl", "Gender": "gender_encoder.pkl",
                  "Sibling": "sibling_encoder.pkl", "Embarked": "embarked_encoder.pkl"}
    for name, file in file_names.items():
        with open(OUT / file, "wb") as handle:
            pickle.dump(encoders[name], handle)
    with open(OUT / "svc_trained_model.pkl", "wb") as handle:
        pickle.dump(svc, handle)

    # --- approach 2: one Pipeline (Exercise 1); unknown category -> -1 (Exercise 2) ----
    pipeline = Pipeline([
        ("encode", OrdinalEncoder(categories=[list(encoders[n].classes_) for n in FEATURES],
                                  handle_unknown="use_encoded_value", unknown_value=-1)),
        ("svc", SVC(**SVC_SETTINGS)),
    ]).fit(frame[FEATURES].iloc[idx_train], y[idx_train])
    joblib.dump(pipeline, OUT / "titanic_pipeline.joblib")

    # The two routes must give the same answer for every possible request.
    everything = pd.MultiIndex.from_product(
        [list(encoders[n].classes_) for n in FEATURES], names=FEATURES).to_frame(index=False)
    every_codes = pd.DataFrame({n: encoders[n].transform(everything[n]) for n in FEATURES})
    same = int((svc.predict(every_codes) == pipeline.predict(everything)).sum())
    print(f"Pickles vs pipeline agree on {same} of {len(everything)} possible requests")

    summary = {
        "rows": int(len(frame)),
        "train_rows": int(len(idx_train)),
        "test_rows": int(len(idx_test)),
        "test_accuracy": round(float(accuracy), 4),
        "baseline_accuracy": round(float(baseline), 4),
        "possible_requests": int(len(everything)),
        "pickle_pipeline_agreement": same,
    }
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved artefacts to", OUT)


if __name__ == "__main__":
    main()
