"""Kvalitní kontrola nahrávek — podezřelé označit REVIEW, nikdy nemazat.

Profily: wav cesta se resolve per mluvčí (recording_path).
"""
import math
from pathlib import Path

try:
    import numpy as np
except ImportError:  # quality bez numpy = jen header kontroly
    np = None

from voice_trainer import dataset as ds
from voice_trainer import stt as stt_mod


def read_wav(path):
    """(sample_rate, channels, duration, samples|None) — mono float."""
    import wave
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        data = w.readframes(frames)
    if sw == 2:
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        arr = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        return sr, ch, frames / sr, None
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    return sr, ch, frames / sr, arr


def check_recording(db, cfg, rec):
    """Vrátí (issues, ok). Zavazné = REVIEW flagy; nic se nemaže."""
    q = cfg["quality"]
    issues = []
    path = ds.recording_path(db, cfg, rec)
    if not path.exists():
        return ["missing wav"], False
    sr, ch, dur, samples = read_wav(path)
    if rec["sample_rate"] and sr != rec["sample_rate"]:
        issues.append(f"samplerate {sr} != db {rec['sample_rate']}")
    if ch != 1:
        issues.append(f"channels {ch}")
    if dur < float(q["min_duration"]):
        issues.append(f"too short ({dur:.2f}s)")
    if dur > float(q["max_duration"]):
        issues.append(f"too long ({dur:.1f}s)")
    if samples is not None:
        peak = float(np.max(np.abs(samples)))
        # clip počítáme až po prvních 100 ms — avfoundation dělá startovací
        # full-scale klik (< 1 ms), který není clipping z příliš hlasého vstupu
        clip_samples = samples[int(sr * 0.1):]
        if clip_samples.size == 0:
            clip_samples = samples
        over = int(np.sum(np.abs(clip_samples) >= float(q["clip_peak"])))
        if peak >= float(q["clip_peak"]) and over >= int(q["clip_count"]):
            issues.append(f"clipping ({over} samples @ {peak:.3f})")
        rms = float(np.sqrt(np.mean(samples ** 2)))
        rms_db = 20 * math.log10(max(rms, 1e-10))
        if rms < 1e-4:
            issues.append("near silent")
        elif rms_db < float(q["quiet_rms_db"]):
            issues.append(f"too quiet ({rms_db:.1f} dB RMS)")
        elif rms_db > float(q["loud_rms_db"]):
            issues.append(f"too loud ({rms_db:.1f} dB RMS)")
    utt = db.utterance(rec["utterance_id"])
    if utt is not None and rec["stt_text"]:
        sim = stt_mod.similarity(utt["text"], rec["stt_text"], utt["language"], cfg)
        if sim < float(cfg["recording"]["similarity_threshold"]):
            issues.append(f"stt mismatch (sim {sim:.2f})")
    return issues, not issues


def run_report(db, cfg, speaker_id=None):
    """Quality report: projde PASS i REVIEW nahrávky (nejnovější per věta
    a mluvčí). speaker_id=None → všechny mluvčí."""
    if speaker_id is None:
        recs = db.all(
            "SELECT r.* FROM recordings r JOIN "
            "(SELECT utterance_id, speaker_id, MAX(id) mid FROM recordings "
            " WHERE status IN ('PASS','REVIEW') "
            " GROUP BY utterance_id, speaker_id) g ON g.mid = r.id "
            "ORDER BY r.utterance_id")
    else:
        recs = db.all(
            "SELECT r.* FROM recordings r JOIN "
            "(SELECT utterance_id, MAX(id) mid FROM recordings "
            " WHERE status IN ('PASS','REVIEW') AND speaker_id=? "
            " GROUP BY utterance_id) g ON g.mid = r.id "
            "ORDER BY r.utterance_id", (speaker_id,))
    rows = []
    for rec in recs:
        path = ds.recording_path(db, cfg, rec)
        if not path.exists():
            sp = db.speaker(rec["speaker_id"]) if rec["speaker_id"] else None
            rows.append({"id": rec["id"], "utterance": rec["utterance_id"],
                         "speaker": sp["name"] if sp else None,
                         "issues": ["missing wav"]})
            continue
        rec2 = dict(rec)
        issues, _ok = check_recording(db, cfg, rec2)
        sp = db.speaker(rec["speaker_id"]) if rec["speaker_id"] else None
        rows.append({"id": rec["id"], "utterance": rec["utterance_id"],
                     "speaker": sp["name"] if sp else None,
                     "issues": issues})
    return rows