"""Train reproducible experimental models directly from the supplied ZIP files.

No archive is extracted. Splits happen at recording-file / machine-ID level.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__:
    from .preprocessing import (AUDIO_FEATURE_VERSION, AUDIO_SECONDS, TARGET_AUDIO_RATE,
                                VIBRATION_FEATURE_VERSION, audio_features, vibration_features)
else:
    from preprocessing import (AUDIO_FEATURE_VERSION, AUDIO_SECONDS, TARGET_AUDIO_RATE,
                               VIBRATION_FEATURE_VERSION, audio_features, vibration_features)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REPORTS = ROOT / "evaluation"
RANDOM_SEED = 42


def wav_from_zip(zf: zipfile.ZipFile, name: str) -> tuple[int, np.ndarray]:
    rate, data = wavfile.read(io.BytesIO(zf.read(name)))
    if data.dtype.kind in "iu":
        info = np.iinfo(data.dtype)
        data = data.astype(np.float32) / max(abs(info.min), abs(info.max))
    else:
        data = data.astype(np.float32)
    return int(rate), data


def choose_audio_files(zf: zipfile.ZipFile, limit: int) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Balance within each valve ID, then hold out whole recordings stratified by class."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name in zf.namelist():
        parts = name.split("/")
        if len(parts) == 4 and parts[-1].endswith(".wav") and parts[-2] in {"normal", "abnormal"}:
            groups[(parts[1], parts[2])].append(name)
    rows = []
    for (machine_id, label), names in sorted(groups.items()):
        rng = random.Random(f"{RANDOM_SEED}:{machine_id}:{label}")
        names.sort()
        rng.shuffle(names)
        names = names[:limit]
        rows.extend((name, int(label == "abnormal"), machine_id) for name in names)
    if not rows:
        raise ValueError("Audio ZIP must contain valve/id_00.../id_06/{normal,abnormal} WAV files")
    labels = [row[1] for row in rows]
    train, test = train_test_split(rows, test_size=.25, stratify=labels,
                                   random_state=RANDOM_SEED)
    return list(train), list(test)


def train_audio(path: Path, limit: int) -> dict:
    with zipfile.ZipFile(path) as zf:
        train_rows, test_rows = choose_audio_files(zf, limit)
        def matrix(rows):
            xs, ys = [], []
            for idx, (name, label, _group) in enumerate(rows, 1):
                rate, audio = wav_from_zip(zf, name)
                xs.append(audio_features(audio, rate))
                ys.append(label)
                if idx % 100 == 0:
                    print(f"  valve audio features: {idx}/{len(rows)}")
            return np.stack(xs), np.asarray(ys)
        X_train, y_train = matrix(train_rows)
        X_test, y_test = matrix(test_rows)
    fit_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=.2,
                                        stratify=y_train, random_state=RANDOM_SEED)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   class_weight="balanced", n_jobs=-1,
                                   random_state=RANDOM_SEED)
    model.fit(X_train[fit_idx], y_train[fit_idx])
    val_probability = model.predict_proba(X_train[val_idx])[:, list(model.classes_).index(1)]
    val_pred = model.predict(X_train[val_idx])
    probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    pred = model.predict(X_test)
    metrics = {"validation": classification_metrics(y_train[val_idx], val_pred, val_probability),
               "test": classification_metrics(y_test, pred, probability)}
    bundle = {"model": model, "feature_version": AUDIO_FEATURE_VERSION,
              "sample_rate": TARGET_AUDIO_RATE, "duration_seconds": AUDIO_SECONDS,
              "labels": {0: "normal", 1: "abnormal"},
              "dataset": "MIMII 6 dB valve; anomaly labels include multiple fault types",
              "split": "recording-level stratified holdout across valve IDs",
              "metrics": metrics}
    joblib.dump(bundle, MODELS / "valve_audio_anomaly.joblib", compress=3)
    return {"name": "Valve audio anomaly (not leak-specific)", "metrics": metrics,
            "train_recordings": len(y_train), "test_recordings": len(y_test),
            "train_labels": dict(Counter(y_train.tolist())),
            "test_labels": dict(Counter(y_test.tolist())),
            "test_machine_ids": sorted({row[2] for row in test_rows}),
            "split": "recording-level stratified holdout; same valve IDs occur in train/test"}


def csv_vibration_from_zip(zf: zipfile.ZipFile, name: str) -> tuple[np.ndarray, float]:
    data = np.loadtxt(io.BytesIO(zf.read(name)), delimiter=",", dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 8:
        raise ValueError(f"Expected 8 columns in {name}; got shape {data.shape}")
    # MAFAULDA: tachometer, underhang X/Y/Z, overhang X/Y/Z, microphone.
    # Use the underhang triaxial accelerometer to match device-motion axes.
    return data[:, 1:4], 50_000.0


def csv_microphone_from_zip(zf: zipfile.ZipFile, name: str) -> tuple[np.ndarray, float]:
    data = np.loadtxt(io.BytesIO(zf.read(name)), delimiter=",", dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 8:
        raise ValueError(f"Expected 8 columns in {name}; got shape {data.shape}")
    # The eighth MAFAULDA channel is the recording microphone (50 kHz).
    return data[:, 7], 50_000


def choose_vibration_files(normal_zip: zipfile.ZipFile, fault_zip: zipfile.ZipFile,
                           per_class_limit: int) -> list[tuple[str, int, str]]:
    normal = sorted(n for n in normal_zip.namelist() if n.endswith(".csv"))
    faults_by_severity: dict[str, list[str]] = defaultdict(list)
    for name in fault_zip.namelist():
        if name.endswith(".csv"):
            severity = name.split("/")[1]
            faults_by_severity[severity].append(name)
    rng = random.Random(RANDOM_SEED)
    # Retain coverage of each imbalance mass while respecting the configured cap.
    selected_faults = []
    severities = sorted(faults_by_severity, key=lambda s: float(s.removesuffix("g")))
    quotas = max(1, per_class_limit // max(1, len(severities)))
    for severity in severities:
        options = sorted(faults_by_severity[severity])
        rng.shuffle(options)
        selected_faults.extend((name, 1, f"fault:{severity}") for name in options[:quotas])
    if len(selected_faults) < per_class_limit:
        already = {item[0] for item in selected_faults}
        rest = [name for names in faults_by_severity.values() for name in names if name not in already]
        rng.shuffle(rest)
        selected_faults.extend((name, 1, f"fault:{name.split('/')[1]}")
                               for name in rest[:per_class_limit - len(selected_faults)])
    rng.shuffle(normal)
    normal = normal[:per_class_limit]
    all_rows = [(n, 0, "normal") for n in normal] + selected_faults[:per_class_limit]
    return sorted(all_rows)


def grouped_vibration_splits(rows: list[tuple[str, int, str]]):
    """Shared file-level partitions keep audio and vibration evaluation aligned."""
    groups = np.asarray([f"{y}:{name}" for name, y, _ in rows])
    y = np.asarray([label for _, label, _ in rows])
    train_val_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2,
        random_state=RANDOM_SEED).split(rows, y, groups))
    train_rel, val_rel = next(GroupShuffleSplit(n_splits=1, test_size=.2,
        random_state=RANDOM_SEED + 1).split(np.zeros(len(train_val_idx)), y[train_val_idx], groups[train_val_idx]))
    return y, train_val_idx[train_rel], train_val_idx[val_rel], test_idx


def train_vibration(normal_path: Path, fault_path: Path, limit: int) -> dict:
    with zipfile.ZipFile(normal_path) as nz, zipfile.ZipFile(fault_path) as fz:
        rows = choose_vibration_files(nz, fz, limit)
        y, train_idx, val_idx, test_idx = grouped_vibration_splits(rows)
        X = []
        for i, (name, _label, _group) in enumerate(rows, 1):
            archive = nz if name.startswith("normal/") else fz
            signal_data, rate = csv_vibration_from_zip(archive, name)
            X.append(vibration_features(signal_data, rate))
            if i % 10 == 0:
                print(f"  vibration recording features: {i}/{len(rows)}")
        X = np.stack(X)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=350, min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_SEED))
    model.fit(X[train_idx], y[train_idx])
    val_pred = model.predict(X[val_idx])
    val_probabilities = model.predict_proba(X[val_idx])[:, list(model[-1].classes_).index(1)]
    pred = model.predict(X[test_idx])
    probabilities = model.predict_proba(X[test_idx])[:, list(model[-1].classes_).index(1)]
    metrics = {"validation": classification_metrics(y[val_idx], val_pred, val_probabilities),
               "test": classification_metrics(y[test_idx], pred, probabilities)}
    train_groups = [rows[i][2] for i in train_idx]
    bundle = {"model": model, "feature_version": VIBRATION_FEATURE_VERSION,
              "sample_rate_training_hz": 50_000,
              "axes": "MAFAULDA underhang axial/radial/tangential accelerometers",
              "labels": {0: "normal", 1: "imbalance"},
              "dataset": "MAFAULDA normal and rotor imbalance; lab accelerometers",
              "metrics": metrics}
    joblib.dump(bundle, MODELS / "vibration_imbalance.joblib", compress=3)
    return {"name": "MAFAULDA normal vs imbalance vibration", "metrics": metrics,
            "train_recordings": len(train_idx), "test_recordings": len(test_idx),
            "train_labels": dict(Counter(y[train_idx].tolist())),
            "test_labels": dict(Counter(y[test_idx].tolist())),
            "severity_groups_in_train": sorted(set(train_groups))}


def train_machine_audio(normal_path: Path, fault_path: Path, limit: int) -> dict:
    """Train machine-health audio from MAFAULDA microphone channels."""
    with zipfile.ZipFile(normal_path) as nz, zipfile.ZipFile(fault_path) as fz:
        rows = choose_vibration_files(nz, fz, limit)
        y, train_idx, val_idx, test_idx = grouped_vibration_splits(rows)
        X = []
        for i, (name, _label, _group) in enumerate(rows, 1):
            archive = nz if name.startswith("normal/") else fz
            signal, rate = csv_microphone_from_zip(archive, name)
            X.append(audio_features(signal, rate))
            if i % 10 == 0:
                print(f"  MAFAULDA microphone features: {i}/{len(rows)}")
        X = np.stack(X)
    model = RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
    model.fit(X[train_idx], y[train_idx])
    val_prob = model.predict_proba(X[val_idx])[:, list(model.classes_).index(1)]
    test_prob = model.predict_proba(X[test_idx])[:, list(model.classes_).index(1)]
    metrics = {"validation": classification_metrics(y[val_idx], model.predict(X[val_idx]), val_prob),
               "test": classification_metrics(y[test_idx], model.predict(X[test_idx]), test_prob)}
    bundle = {"model": model, "feature_version": AUDIO_FEATURE_VERSION,
        "sample_rate_training_hz": 50_000, "feature_sample_rate_hz": TARGET_AUDIO_RATE,
        "duration_seconds": AUDIO_SECONDS, "labels": {0: "normal", 1: "imbalance"},
        "dataset": "MAFAULDA recording microphone; normal vs rotor imbalance",
        "split": "recording-level partitions shared with vibration model", "metrics": metrics}
    joblib.dump(bundle, MODELS / "machine_audio_imbalance.joblib", compress=3)
    return {"name": "MAFAULDA normal vs imbalance microphone audio", "metrics": metrics,
            "train_recordings": len(train_idx), "validation_recordings": len(val_idx),
            "test_recordings": len(test_idx), "test_labels": dict(Counter(y[test_idx].tolist())),
            "split": "same recording-level train/validation/test partitions as vibration model"}


def train_energy(path: Path) -> dict:
    """Experimental steel dataset load-type classifier; not an idle detector."""
    with zipfile.ZipFile(path) as zf:
        csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if not csv_name:
            raise ValueError("Steel ZIP contains no CSV")
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "Load_Type", "Usage_kWh"]).sort_values("date")
    df["hour"] = df["date"].dt.hour + df["date"].dt.minute / 60.0
    df["month"] = df["date"].dt.month
    target = "Load_Type"
    numeric = ["Usage_kWh", "Lagging_Current_Reactive.Power_kVarh",
               "Leading_Current_Reactive_Power_kVarh", "CO2(tCO2)",
               "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor", "NSM", "hour", "month"]
    categorical = ["WeekStatus", "Day_of_week"]
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    for col in numeric:
        df[col] = df[col].fillna(df[col].median())
    train_end, val_end = int(len(df) * .7), int(len(df) * .85)
    train, val, test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    transform = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = make_pipeline(transform, RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED))
    model.fit(train[numeric + categorical], train[target])
    val_pred = model.predict(val[numeric + categorical])
    pred = model.predict(test[numeric + categorical])
    labels = sorted(df[target].unique())
    def load_metrics(actual, predicted):
        return {"accuracy": float(accuracy_score(actual, predicted)),
                "macro_precision": float(precision_score(actual, predicted, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(actual, predicted, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
                "confusion_matrix_labels": labels,
                "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
                "period_start": str(df.loc[actual.index, "date"].min()),
                "period_end": str(df.loc[actual.index, "date"].max())}
    metrics = {"validation": load_metrics(val[target], val_pred),
               "test": load_metrics(test[target], pred)}
    bundle = {"model": model, "numeric_features": numeric, "categorical_features": categorical,
              "labels": labels, "metrics": metrics,
              "dataset": "Steel industry energy; load class prediction (not idle/standby)"}
    joblib.dump(bundle, MODELS / "steel_load_type.joblib", compress=3)
    # Forecast the following 15-minute interval from past Usage_kWh only.
    series = df[["date", "Usage_kWh"]].copy().sort_values("date")
    series["lag_15m"] = series["Usage_kWh"].shift(1)
    series["lag_1h"] = series["Usage_kWh"].shift(4)
    series["lag_24h"] = series["Usage_kWh"].shift(96)
    series["previous_hour_mean"] = series["Usage_kWh"].shift(1).rolling(4).mean()
    series["previous_day_mean"] = series["Usage_kWh"].shift(1).rolling(96).mean()
    series["target_kwh"] = series["Usage_kWh"].shift(-1)
    series["target_hour"] = series["date"].shift(-1).dt.hour + series["date"].shift(-1).dt.minute / 60
    series["target_weekday"] = series["date"].shift(-1).dt.dayofweek
    series = series.dropna()
    forecast_features = ["lag_15m", "lag_1h", "lag_24h", "previous_hour_mean",
                         "previous_day_mean", "target_hour", "target_weekday"]
    forecast_train_end, forecast_val_end = int(len(series) * .7), int(len(series) * .85)
    forecast_train = series.iloc[:forecast_train_end]
    forecast_val, forecast_test = series.iloc[forecast_train_end:forecast_val_end], series.iloc[forecast_val_end:]
    forecast_model = RandomForestRegressor(n_estimators=120, max_depth=16,
        min_samples_leaf=4, max_features=.9, n_jobs=-1, random_state=RANDOM_SEED)
    forecast_model.fit(forecast_train[forecast_features], forecast_train["target_kwh"])
    def forecast_metrics_for(subset):
        prediction = forecast_model.predict(subset[forecast_features])
        truth = subset["target_kwh"].to_numpy()
        return {"mae_kwh": float(mean_absolute_error(truth, prediction)),
            "rmse_kwh": float(np.sqrt(np.mean((truth - prediction) ** 2))),
            "r2": float(r2_score(truth, prediction)),
            "mape_percent": float(np.mean(np.abs((truth - prediction) / np.maximum(np.abs(truth), .1))) * 100),
            "rows": len(subset), "period_start": str(subset["date"].min()),
            "period_end": str(subset["date"].max())}
    forecast_metrics = {"validation": forecast_metrics_for(forecast_val),
                        "test": forecast_metrics_for(forecast_test)}
    joblib.dump({"model": forecast_model, "features": forecast_features,
        "horizon_minutes": 15, "history_interval_minutes": 15,
        "metrics": forecast_metrics, "dataset": "UCI steel usage; prior quarter-hour history"},
        MODELS / "steel_energy_forecast.joblib", compress=3)
    return {"name": "Steel load type (not idle detector) and 15-minute energy forecast",
            "metrics": metrics, "forecast_metrics": forecast_metrics,
            "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
            "labels": df[target].value_counts().to_dict()}


def classification_metrics(y_true, pred, probability) -> dict:
    labels = [0, 1]
    result = {"accuracy": float(accuracy_score(y_true, pred)),
              "precision_fault": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
              "recall_fault": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
              "f1_fault": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
              "confusion_matrix_labels": ["normal", "fault"],
              "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist()}
    matrix = np.asarray(result["confusion_matrix"])
    result["false_positives"] = int(matrix[0, 1])
    result["false_negatives"] = int(matrix[1, 0])
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
        result["pr_auc_average_precision"] = float(average_precision_score(y_true, probability))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-zip", type=Path, required=True)
    parser.add_argument("--imbalance-zip", type=Path, required=True)
    parser.add_argument("--normal-zip", type=Path, required=True)
    parser.add_argument("--energy-zip", type=Path, required=True)
    parser.add_argument("--audio-files-per-label-machine", type=int, default=100)
    parser.add_argument("--vibration-files-per-class", type=int, default=49)
    parser.add_argument("--only-energy", action="store_true",
                        help="train the small steel load and usage models without reading the multi-GB signal archives")
    parser.add_argument("--only-machine-audio", action="store_true",
                        help="train only the MAFAULDA microphone model")
    args = parser.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    print("Training from supplied ZIPs without extracting them. Recording-level splits; seed=42.")
    results = [train_energy(args.energy_zip)] if args.only_energy else (
        [train_machine_audio(args.normal_zip, args.imbalance_zip, args.vibration_files_per_class)]
        if args.only_machine_audio else [
               train_audio(args.audio_zip, args.audio_files_per_label_machine),
               train_vibration(args.normal_zip, args.imbalance_zip, args.vibration_files_per_class),
               train_machine_audio(args.normal_zip, args.imbalance_zip, args.vibration_files_per_class),
               train_energy(args.energy_zip)])
    report = {"trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "random_seed": RANDOM_SEED,
              "models": results,
              "limitations": [
                  "Initial cross-machine holdout result is preserved separately in early_cross_machine_audio_diagnostic.json.",
                  "MIMII valve anomaly labels are not leak-specific.",
                  "MAFAULDA lab accelerometers are not equivalent to smartphone motion sensors.",
                  "The steel dataset predicts its own Light/Medium/Maximum load labels; it is not idle/standby ground truth.",
                  "No electrical fault labels were supplied; live electrical checks remain meter-versus-configured-baseline rules.",
              ]}
    out = REPORTS / ("energy_training_metrics.json" if args.only_energy else "machine_audio_training_metrics.json" if args.only_machine_audio else "training_metrics.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for item in results:
        print(f"\n{item['name']}\n{json.dumps(item, indent=2, default=str)}")
    print(f"\nSaved metrics: {out}")


if __name__ == "__main__":
    main()
