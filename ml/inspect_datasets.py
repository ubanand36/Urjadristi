"""Inspect the four supplied archives without extracting them."""
from __future__ import annotations
import argparse, collections, csv, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile


def inspect_audio(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        wavs = [n for n in zf.namelist() if n.lower().endswith('.wav')]
        counts = collections.Counter('/'.join(n.split('/')[:3]) for n in wavs)
        chosen = next((n for n in wavs if '/normal/' in n), wavs[0])
        rate, x = wavfile.read(io.BytesIO(zf.read(chosen)))
        return {"archive": str(path), "wav_count": len(wavs), "uncompressed_bytes": sum(i.file_size for i in zf.infolist()),
                "groups_machine_and_label": dict(counts), "sample_file": chosen,
                "sample_rate_hz": rate, "channels": 1 if x.ndim == 1 else x.shape[1],
                "duration_seconds": len(x) / rate, "dtype": str(x.dtype),
                "label_note": "normal/abnormal labels; abnormal does not identify a fault subtype"}


def inspect_vibration(paths: list[Path]) -> dict:
    archives = []
    for path in paths:
        with zipfile.ZipFile(path) as zf:
            files = [n for n in zf.namelist() if n.lower().endswith('.csv')]
            sample = files[0]
            with zf.open(sample) as f:
                sample_lines = [f.readline().decode('utf-8', 'replace').strip() for _ in range(10)]
            parsed = np.asarray([[float(v) for v in row.split(',')] for row in sample_lines if row])
            archives.append({"archive": str(path), "csv_files": len(files),
                "group_counts": dict(collections.Counter(n.split('/')[1] if '/' in n else 'unknown' for n in files)),
                "sample_file": sample, "sample_columns": parsed.shape[1],
                "sample_rows_checked": len(parsed), "sample_missing_values": int(np.isnan(parsed).sum()),
                "sample_exact_duplicate_rows": int(len(parsed) - len(np.unique(parsed, axis=0))),
                "sampling_rate_hz": 50_000, "duration_seconds_per_capture": 5,
                "column_order": ["tachometer", "underhang_axial", "underhang_radial", "underhang_tangential",
                                 "overhang_axial", "overhang_radial", "overhang_tangential", "microphone"],
                "header": False, "units": "sensor acquisition units; accelerometer values are not smartphone SI readings"})
    return {"archives": archives, "row_count_method": "250,000 rows per complete MAFAULDA recording; sampled preview is shown above"}


def inspect_energy(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith('.csv'))
        with zf.open(name) as f:
            df = pd.read_csv(f, encoding='utf-8-sig')
    dupes = int(df.duplicated().sum())
    missing = {k: int(v) for k, v in df.isna().sum().items() if v}
    return {"archive": str(path), "csv_file": name, "rows": len(df), "columns": list(df.columns),
            "missing_by_column": missing, "duplicate_rows": dupes,
            "load_type_counts": df['Load_Type'].value_counts(dropna=False).to_dict(),
            "date_range": [str(df['date'].iloc[0]), str(df['date'].iloc[-1])],
            "target_note": "Light/Medium/Maximum industrial load labels; not idle/standby labels"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio-zip', type=Path, required=True)
    p.add_argument('--imbalance-zip', type=Path, required=True)
    p.add_argument('--normal-zip', type=Path, required=True)
    p.add_argument('--energy-zip', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    summary = {"audio": inspect_audio(a.audio_zip),
               "vibration": inspect_vibration([a.imbalance_zip, a.normal_zip]),
               "energy": inspect_energy(a.energy_zip),
               "dataset_limitations": [
                   "Valve audio provides only normal/abnormal labels, not leak-specific ground truth.",
                   "MAFAULDA captures are from a 50 kHz laboratory rig; phone motion sensors have different rate, placement and units.",
                   "Steel energy labels indicate load level, not machine idle/standby or electrical faults.",
                   "No missing/duplicate scan of every row in the multi-gigabyte CSV archives was performed by this quick inspector; sampled rows are explicitly counted above."]}
    report = json.dumps(summary, indent=2, default=str)
    print(report)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(report, encoding='utf-8')

if __name__ == '__main__':
    main()
