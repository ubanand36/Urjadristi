"""FastAPI inference service for UrjaDrishti's trained experimental models."""
from __future__ import annotations
import base64, binascii, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
import joblib
import numpy as np
import jwt
from jwt import PyJWKClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile
from scipy.io import wavfile

from ml.preprocessing import (AUDIO_FEATURE_VERSION, VIBRATION_FEATURE_VERSION,
                              audio_features, decode_wav, vibration_features)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv('URJA_MODEL_DIR', str(ROOT / 'models')))
app = FastAPI(title="UrjaDrishti ML Inference", version="0.1.0")
origins = [v.strip() for v in os.getenv('URJA_CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if v.strip()]
FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID', 'urjadrishti-a7987')
FIREBASE_KEYS = PyJWKClient('https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com')
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'POST'],
                   allow_headers=['*'])


@app.middleware('http')
async def require_firebase_user(request: Request, call_next):
    if request.url.path.startswith('/api/ml/') and request.method != 'OPTIONS':
        header = request.headers.get('authorization', '')
        if not header.lower().startswith('bearer '):
            return JSONResponse(status_code=401, content={"detail": "Sign in with Firebase before using model inference."})
        try:
            token = header.split(' ', 1)[1].strip()
            key = FIREBASE_KEYS.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=['RS256'], audience=FIREBASE_PROJECT_ID,
                                issuer=f'https://securetoken.google.com/{FIREBASE_PROJECT_ID}',
                                options={"require": ["exp", "iat", "auth_time", "sub", "aud", "iss"]})
            if not claims.get('sub'):
                raise jwt.InvalidTokenError('Firebase user ID is missing')
            request.state.firebase_uid = claims['sub']
        except Exception:
            return JSONResponse(status_code=401, content={"detail": "Firebase sign-in token is invalid or expired. Sign in again."})
    return await call_next(request)


def load_bundle(name: str) -> dict:
    path = MODEL_DIR / name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Required trained model is not installed: {name}")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not load trained model {name}") from exc


def class_score(bundle: dict, x: np.ndarray, fault_class: int = 1) -> float:
    model = bundle['model']
    classes = list(model.classes_ if hasattr(model, 'classes_') else model[-1].classes_)
    if fault_class not in classes:
        raise HTTPException(status_code=503, detail="Trained model is missing the expected fault class")
    return float(model.predict_proba(x.reshape(1, -1))[0, classes.index(fault_class)])


def bounded_score(value: float) -> int:
    return int(np.clip(round(value * 100), 0, 100))


def request_audio(body: dict) -> tuple[int, np.ndarray]:
    try:
        audio_bytes = base64.b64decode(body['audio_wav_base64'], validate=True)
        if len(audio_bytes) > 2_000_000:
            raise HTTPException(status_code=413, detail='Audio payload is too large; record a short 8-second clip.')
        return decode_wav(audio_bytes)
    except (KeyError, ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=422, detail='A valid base64-encoded WAV recording is required') from exc


async def upload_bytes(upload: StarletteUploadFile, description: str) -> bytes:
    content = await upload.read(64 * 1024 * 1024 + 1)
    if len(content) > 64 * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f'{description} file must be 64 MB or smaller.')
    if not content:
        raise HTTPException(status_code=422, detail=f'{description} file is empty.')
    return content


async def uploaded_machine_audio(upload: StarletteUploadFile) -> tuple[int, np.ndarray]:
    content = await upload_bytes(upload, 'Audio')
    suffix = Path(upload.filename or '').suffix.lower()
    try:
        if suffix == '.wav':
            return decode_wav(content)
        if suffix == '.csv':
            data = np.loadtxt(__import__('io').BytesIO(content), delimiter=',', dtype=np.float32)
            if data.ndim != 2 or data.shape[1] != 8:
                raise ValueError('MAFAULDA audio CSV must have 8 headerless numeric columns.')
            return 50_000, data[:, 7]
        raise ValueError('Use a WAV file or an 8-column MAFAULDA CSV.')
    except Exception as exc:
        raise HTTPException(status_code=422, detail='Audio upload must be a valid WAV or an 8-column, headerless MAFAULDA CSV (microphone is column 8).') from exc


async def uploaded_vibration(upload: StarletteUploadFile) -> tuple[np.ndarray, float]:
    content = await upload_bytes(upload, 'Vibration')
    if Path(upload.filename or '').suffix.lower() != '.csv':
        raise HTTPException(status_code=422, detail='Vibration upload must be a headerless numeric CSV file.')
    try:
        data = np.loadtxt(__import__('io').BytesIO(content), delimiter=',', dtype=np.float32)
        if data.ndim != 2 or data.shape[1] not in (3, 8):
            raise ValueError('Expected three XYZ columns or eight MAFAULDA columns.')
        axes = data[:, :3] if data.shape[1] == 3 else data[:, 1:4]
        return axes, 50_000.0
    except Exception as exc:
        raise HTTPException(status_code=422, detail='Vibration CSV must contain headerless XYZ columns or the 8-column MAFAULDA format.') from exc


@app.get('/api/ml/status')
def status():
    expected = ['valve_audio_anomaly.joblib', 'machine_audio_imbalance.joblib',
                'vibration_imbalance.joblib', 'steel_energy_forecast.joblib']
    return {"ready": all((MODEL_DIR / name).is_file() for name in expected),
            "models": {name: (MODEL_DIR / name).is_file() for name in expected}}


@app.post('/api/ml/air-leak')
async def air_leak(request: Request):
    bundle = load_bundle('valve_audio_anomaly.joblib')
    if bundle.get('feature_version') != AUDIO_FEATURE_VERSION:
        raise HTTPException(status_code=503, detail='Audio preprocessing version does not match the trained model')
    try:
        if request.headers.get('content-type', '').lower().startswith('multipart/form-data'):
            form = await request.form()
            machine_id = str(form.get('machine_id', 'unknown'))
            upload = form.get('audio_file')
            if not isinstance(upload, StarletteUploadFile):
                raise HTTPException(status_code=422, detail='A valve WAV audio file is required.')
            if Path(upload.filename or '').suffix.lower() != '.wav':
                raise HTTPException(status_code=422, detail='Air-leak upload must be a WAV file, preferably in the MIMII valve recording format.')
            rate, samples = await uploaded_machine_audio(upload)
        else:
            body = await request.json()
            machine_id = str(body.get('machine_id', 'unknown'))
            rate, samples = request_audio(body)
        score = bounded_score(class_score(bundle, audio_features(samples, rate)))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail='Unable to process this audio recording; record again as WAV audio') from exc
    candidate = score >= 55
    return {"machine_id": machine_id, "leak_score": score,
            "status": "Abnormal Sound Candidate" if candidate else "No Abnormal Sound Flagged",
            "leak_candidate": candidate,
            "possible_reason": "Valve anomaly model flagged sound. This dataset is not leak-specific; inspect the site to confirm." if candidate else "No strong valve-anomaly pattern was flagged; this does not rule out an air leak.",
            "model_scope": "general valve sound anomaly, not a calibrated air-leak probability",
            "timestamp": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}


@app.post('/api/ml/energy-forecast')
async def energy_forecast(payload: dict):
    """Predict next quarter-hour kWh from the previous 24 hours of 15-minute usage."""
    bundle = load_bundle('steel_energy_forecast.joblib')
    try:
        history = np.asarray(payload['history_usage_kwh'], dtype=float)
        if len(history) < 96 or not np.all(np.isfinite(history[-96:])) or np.any(history[-96:] < 0):
            raise ValueError('96 valid nonnegative quarter-hour readings are required')
        target_time = datetime.fromisoformat(payload['next_timestamp'].replace('Z', '+00:00'))
        values = {
            'lag_15m': history[-1], 'lag_1h': history[-4], 'lag_24h': history[-96],
            'previous_hour_mean': float(np.mean(history[-4:])),
            'previous_day_mean': float(np.mean(history[-96:])),
            'target_hour': target_time.hour + target_time.minute / 60,
            'target_weekday': target_time.weekday(),
        }
        vector = np.asarray([[values[name] for name in bundle['features']]], dtype=float)
        prediction = float(bundle['model'].predict(vector)[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail='Provide 96 ordered 15-minute kWh readings and next_timestamp (ISO-8601).') from exc
    return {"predicted_usage_kwh": max(0.0, prediction),
            "interval_minutes": 15,
            "next_timestamp": target_time.isoformat(),
            "model_scope": "experimental 15-minute aggregate steel-industry demand forecast; not a per-machine fault/idle model",
            "metrics": bundle['metrics']}


@app.post('/api/ml/machine-health')
async def machine_health(request: Request):
    audio_bundle = load_bundle('machine_audio_imbalance.joblib')
    vibration_bundle = load_bundle('vibration_imbalance.joblib')
    if audio_bundle.get('feature_version') != AUDIO_FEATURE_VERSION or vibration_bundle.get('feature_version') != VIBRATION_FEATURE_VERSION:
        raise HTTPException(status_code=503, detail='Model/preprocessing version mismatch; retrain the model bundle')
    try:
        is_multipart = request.headers.get('content-type', '').lower().startswith('multipart/form-data')
        if is_multipart:
            form = await request.form()
            machine_id = str(form['machine_id'])
            current_power_kw = float(form['current_power_kw'])
            normal_power_kw = float(form['normal_power_kw'])
            audio_upload = form.get('audio_file')
            vibration_upload = form.get('vibration_file')
            if not isinstance(audio_upload, StarletteUploadFile):
                raise ValueError('An audio file is required.')
            rate, waveform = await uploaded_machine_audio(audio_upload)
            if isinstance(vibration_upload, StarletteUploadFile):
                samples, sample_rate = await uploaded_vibration(vibration_upload)
            else:
                payload = json.loads(str(form['vibration_samples']))
                raw_samples = np.asarray(payload['samples'], dtype=np.float32)
                if raw_samples.ndim != 2 or raw_samples.shape[1] not in (3, 4):
                    raise ValueError('Expected timestamped XYZ values or XYZ triples.')
                if raw_samples.shape[1] == 4:
                    elapsed_ms = float(raw_samples[-1, 0] - raw_samples[0, 0])
                    sample_rate = (len(raw_samples) - 1) * 1000.0 / elapsed_ms
                    samples = raw_samples[:, 1:4]
                else:
                    sample_rate = float(payload['sample_rate_hz'])
                    samples = raw_samples
        else:
            body = await request.json()
            machine_id = str(body['machine_id'])
            current_power_kw = float(body['current_power_kw'])
            normal_power_kw = float(body['normal_power_kw'])
            rate, waveform = request_audio(body)
            payload = body['vibration']
            raw_samples = np.asarray(payload['samples'], dtype=np.float32)
            if raw_samples.ndim != 2 or raw_samples.shape[1] not in (3, 4):
                raise ValueError('Expected timestamped XYZ values or XYZ triples')
            if raw_samples.shape[1] == 4:
                elapsed_ms = float(raw_samples[-1, 0] - raw_samples[0, 0])
                sample_rate = (len(raw_samples) - 1) * 1000.0 / elapsed_ms
                samples = raw_samples[:, 1:4]
            else:
                sample_rate = float(payload['sample_rate_hz'])
                samples = raw_samples
        if current_power_kw < 0 or normal_power_kw <= 0:
            raise ValueError('Valid nonnegative meter and positive baseline readings are required')
        audio_score = class_score(audio_bundle, audio_features(waveform, rate))
        vib_score = class_score(vibration_bundle, vibration_features(samples, sample_rate))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail='Audio, vibration, or meter data could not be processed. Record again and check sensor permissions.') from exc
    electrical_score = min(1.0, abs(current_power_kw - normal_power_kw) / normal_power_kw)
    anomaly = .40 * electrical_score + .35 * vib_score + .25 * audio_score
    health = 100 - bounded_score(anomaly)
    state = 'Healthy' if health >= 80 else 'Attention' if health >= 60 else 'Possible Fault'
    evidence = max([(audio_score, 'Machine audio pattern differs from MAFAULDA imbalance recordings'),
                    (vib_score, 'Vibration pattern differs from the imbalance training recordings'),
                    (electrical_score, 'Power reading differs from the configured operating baseline')])
    return {"machine_id": machine_id, "health_score": health, "status": state,
            "audio_anomaly": bounded_score(audio_score),
            "vibration_anomaly": bounded_score(vib_score),
            "electrical_anomaly": bounded_score(electrical_score),
            "possible_reason": evidence[1],
            "timestamp": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            "model_scope": "experimental lab-trained models; smartphone-to-industrial transfer is unvalidated",
            "score_note": "Random-forest vote fractions are uncalibrated scores; fusion weights and thresholds are prototype assumptions"}
