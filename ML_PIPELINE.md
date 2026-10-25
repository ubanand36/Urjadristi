# UrjaDrishti ML pipeline

This repository now has an offline training pipeline and a separate FastAPI inference service. The existing page styling and navigation are retained. Machine Health and the sound scan send captured sensor recordings to `/api/ml/*`; they do not generate a result when capture, the API, or a trained model is unavailable.

## Dataset inspection and interpretation

The supplied files are read directly from ZIP archives; training does not extract their multi-gigabyte contents. The MIMII dataset groups several malfunction types under its abnormal sound label, including leakage and other faults, so its valve classifier cannot identify a leak as the unique cause ([MIMII paper](https://arxiv.org/abs/1909.09347)). MAFAULDA documents 50 kHz, five-second recordings, six accelerometer channels, a tachometer, a microphone, and 6g–35g imbalance conditions ([UFRJ database page](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html)). UCI describes the steel data labels as Light, Medium, and Maximum load—not idle state ([UCI dataset page](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)).

| Supplied archive | Observed contents | Pipeline use and limitation |
| --- | --- | --- |
| `6_dB_valve.zip` | 4,170 WAV recordings; four valve IDs; 8 channels; 16 kHz; 16-bit PCM; 10 seconds; `normal` / `abnormal` folders | Trains a valve sound anomaly classifier. The abnormal class does not say which fault happened, so this is **not a leak-specific detector**. The user interface calls its output an abnormal-sound candidate and does not claim that no leak exists when it flags none. |
| `imbalance.zip` + `normal.zip` | 333 fault CSVs across 6g–35g imbalance levels and 49 normal CSVs; headerless, 8 numeric columns, 250,000 rows (5 seconds) per capture; 50 kHz | Trains two normal-versus-imbalance classifiers: one using the first three accelerometer channels (underhang axial/radial/tangential) and one using the eighth microphone channel. Both use identical recording-level train/validation/test splits. These are lab sensors, not phone-calibrated inputs; transfer to phone recordings remains unvalidated. |
| `steel+industry+energy+consumption.zip` | `Steel_industry_data.csv`, 35,040 quarter-hour rows with electricity, reactive power, power factor, time/calendar fields, and Light/Medium/Maximum load type | Trains an experimental load-type classifier and next-quarter-hour factory energy forecast from the previous day of usage. The labels do **not** represent idle/standby or machine electrical faults. The live electrical anomaly score is the absolute meter-to-configured-baseline deviation; a compatible labeled factory meter dataset is needed for a learned electrical-fault model. |

The quick inspector checks all archive members and inspects representative signal/CSV samples; it explicitly reports that it does not scan every row in the 6.6 GB vibration CSV data for missing or duplicate values. Training parses the selected complete source recordings and rejects invalid feature rows. Dataset provenance: [MIMII dataset and baseline](https://github.com/MIMII-hitachi/mimii_baseline), [MAFAULDA research database](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html), and [UCI Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption).

## Install, inspect, and train (PowerShell)

From `E:\Urjadristi`:

```powershell
python -m pip install -r ml/requirements.txt
python -m ml.inspect_datasets `
  --audio-zip 'C:\Users\ak059\Downloads\6_dB_valve.zip' `
  --imbalance-zip 'C:\Users\ak059\Downloads\imbalance.zip' `
  --normal-zip 'C:\Users\ak059\Downloads\normal.zip' `
  --energy-zip 'C:\Users\ak059\Downloads\steel+industry+energy+consumption.zip' `
  --output ml/evaluation/dataset_inspection.json
python -m ml.train_models `
  --audio-zip 'C:\Users\ak059\Downloads\6_dB_valve.zip' `
  --imbalance-zip 'C:\Users\ak059\Downloads\imbalance.zip' `
  --normal-zip 'C:\Users\ak059\Downloads\normal.zip' `
  --energy-zip 'C:\Users\ak059\Downloads\steel+industry+energy+consumption.zip'
```

Training writes model bundles to `ml/models/` and measured evaluation metrics to `ml/evaluation/training_metrics.json`. It defaults to up to 100 audio recordings per class per valve ID and up to 49 source files from each vibration class; selected ZIP members are decompressed in memory one at a time. Pass `--audio-files-per-label-machine N` and `--vibration-files-per-class N` to tune runtime. Signal model validation/test splits are separated by recording file; the MAFAULDA microphone and vibration models use matching partitions. Steel classification and forecast rows are split chronologically (70% train, 15% validation, 15% test). `--only-energy` trains just the small energy datasets and writes `energy_training_metrics.json`; `--only-machine-audio` retrains the MAFAULDA microphone model and writes `machine_audio_training_metrics.json`. The initial cross-machine valve-audio experiment failed to detect abnormal clips from valve `id_06` (zero fault recall); its metrics are preserved in `ml/evaluation/early_cross_machine_audio_diagnostic.json`. A later recording-level split across IDs produced 93.3% test accuracy / 93.2% fault F1 on 120 clips, but shares all valve IDs between training and test. That does **not** establish cross-machine transfer. The vibration and MAFAULDA microphone test splits contain only 20 recordings each. These scores are experimental, not production accuracy guarantees.

## Training run recorded in this workspace

The saved valve bundle used 60 recordings per label for each of four IDs (480 total). Its file-level split test was 93.3% accuracy, 93.2% fault F1, 0.984 ROC-AUC, with 3 false positives and 5 false negatives. The separate `id_06`-only holdout had 0% fault recall; this demonstrates strong domain-transfer risk. The vibration bundle used 49 normal and 49 imbalance recordings (98 total); on a 20-recording test split it reached 90% accuracy and 87.5% fault F1 (one false positive and one false negative). The microphone model, trained on the eighth MAFAULDA channel and evaluated on the same 20 held-out recordings, reached 95% test accuracy / 93.3% fault F1 (zero false positives, one false negative); its 16-recording validation split reached 68.8% accuracy / 70.6% fault F1. These small evaluations do not establish phone-sensor performance. Details are in `machine_audio_training_metrics.json`. The steel load-type model's final chronological test reached 65.8% accuracy / 56.1% macro F1. The 15-minute kWh forecaster reached MAE 5.22 kWh and R² 0.892 on the chronological test; MAPE was 35.9%, so relative error is high where actual usage is small. Full confusion matrices and periods are in the evaluation JSON files.

## Run locally

In a terminal, install Python dependencies as above, then start the API:

```powershell
npm run ml:api
```

In another terminal, run the existing website:

```powershell
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`. The API endpoints are:

- `POST /api/ml/machine-health` — multipart WAV or MAFAULDA CSV audio, plus either a headerless vibration CSV or timestamped 3-axis motion samples, machine ID, current kW and configured normal kW; returns component anomaly scores and a prototype fused health score. Files are processed for that request and are not stored by this API.
- `POST /api/ml/air-leak` — multipart WAV upload (`audio_file`) or legacy base64 WAV audio, plus machine/location ID; returns an abnormal-valve-sound candidate score, not a leak probability.
- `POST /api/ml/energy-forecast` — 96 ordered quarter-hour kWh values plus the next interval time; returns an experimental aggregate factory usage estimate. The current website has no meter-history input to call this endpoint yet.
- `GET /api/ml/status` — reports whether required trained model files are available.

All `/api/ml/*` endpoints require a Firebase Authentication ID token in `Authorization: Bearer <token>`. The API checks its signature, project audience, and issuer against Google's published signing keys. The website sends the current signed-in user's ID token. Set `FIREBASE_PROJECT_ID` when using a different Firebase project. Set `URJA_CORS_ORIGINS` to a comma-separated list of allowed website origins.

Machine-health audio upload accepts WAV files or headerless MAFAULDA CSVs with eight numeric columns (the microphone signal is column 8, at 50 kHz). Air-leak upload accepts WAV valve recordings, ideally matching MIMII audio (16 kHz, 8 channels); multichannel sound is downmixed to mono for inference. Vibration upload accepts headerless CSVs with either three XYZ columns or the eight-column MAFAULDA structure (XYZ are columns 2–4); CSV vibration inputs are interpreted as 50 kHz. The service applies the same audio and vibration feature extraction used during model training. Uploaded files are sent to the configured inference service, limited to 64 MB each, processed in memory, and not saved by this API. Recorded data continues to use the browser microphone and device-motion preprocessing; unsupported sensors, missing models, or API errors stop the prediction instead of substituting fake readings.

## Deployment

Firebase Hosting serves only the static frontend in this project; it does not run this Python API or the model files. Deploy the FastAPI service and trained model artifacts to a Python-capable backend first. Then set `window.URJA_ML_API_URL` in `public/ml-config.js` to that HTTPS service origin, allow the website origin in `URJA_CORS_ORIGINS`, and build/deploy Hosting. Firebase token verification is enabled; add rate limiting before public production use. Firebase Hosting has not been configured to rewrite `/api` to an unspecified service.

## Known limitations

- The MIMII abnormal label includes different valve faults; it cannot teach leak/no-leak without leak-specific labeled data.
- Smartphone microphone characteristics differ from the eight-microphone training array.
- MAFAULDA's 50 kHz mounted accelerometers differ in rate, mounting, and units from phone motion sensors. Scores on phones are experimental and need a representative labeled phone dataset and external validation.
- Existing demo meter values are not a connected industrial meter feed. Electrical deviation scoring only becomes meaningful when `current_power_kw` and the baseline are real readings.
- The source dataset model scores and hand-selected fusion weights/thresholds are not calibrated probabilities or certified safety thresholds.
- The cross-machine audio diagnostic failed: 0% fault recall for unseen valve ID `id_06`. The file-level test had 3 false positives and 5 false negatives; its metrics do not establish transfer to a new machine ID.
- The steel dataset's load labels are not idle/standby labels; the existing idle rule remains a meter-threshold/duration rule.
- User factory readings remain in browser storage as before; this API does not sync application records to Firebase.
- Raw microphone and motion samples are processed in request memory and are not stored for later fine-tuning. Building a consented, verified UrjaDrishti dataset remains future work.
