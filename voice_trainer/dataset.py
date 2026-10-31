"""Dataset operace — metadata.csv pro Piper, build cumulative dataset_vN.

Profily mluvčích (2026-09-26): WAV cesty jsou dataset/wav/<mluvčí>/NNNNNN.wav
(stejná věta nahráná dvěma mluvčími = dva soubory, žádná kolize názvů).
"""
import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

try:
    import numpy as np
except ImportError:  # bez numpy se hlasitost nesrovnává
    np = None

PEAK_LIMIT_DB = -1.0
MAX_GAIN_DB = 40.0


def wav_dir(cfg, speaker=None):
    d = Path(cfg["data_root"]) / "dataset" / "wav"
    return d / speaker if speaker else d


def master_dir(cfg, speaker=None):
    d = Path(cfg["data_root"]) / "dataset" / "master"
    return d / speaker if speaker else d


def incoming_dir(cfg):
    return Path(cfg["data_root"]) / "incoming"


def training_root(cfg):
    return Path(cfg["data_root"]) / "training"


def utterance_name(uid):
    return f"{uid:06d}.wav"


def base_sample_rate(cfg, base_config=None):
    """Sample rate tréninku — čte se z base checkpoint configu (žádné
    natvrdo hodnoty); fallback jen pro první ukládání bez checkpointu."""
    t = cfg["training"]
    for path in (base_config, t.get("base_config")):
        if path and Path(path).exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                sr = data.get("audio", {}).get("sample_rate") or data.get("sample_rate")
                if sr:
                    return int(sr)
            except (OSError, ValueError, AttributeError):
                pass
    return int(t["fallback_sample_rate"])


def recording_path(db, cfg, rec):
    """Absolutní cesta wav nahrávky — podle speaker_id nahrávky."""
    sp = db.speaker(rec["speaker_id"]) if rec["speaker_id"] else None
    return wav_dir(cfg, sp["name"] if sp else None) / (rec["wav"] or "")


def write_metadata(db, cfg, speaker_id=None):
    """metadata-<mluvčí>.csv = odvozený export PASS nahrávek (source of truth
    je DB). Formát Piper: soubor.wav|text. speaker_id=None → všechny."""
    targets = []
    if speaker_id is not None:
        targets = [speaker_id]
    else:
        targets = [s["id"] for s in db.speakers()]
    total = 0
    for sid in targets:
        sp = db.speaker(sid)
        wdir = wav_dir(cfg, sp["name"])
        wdir.mkdir(parents=True, exist_ok=True)
        out = Path(cfg["data_root"]) / "dataset" / f"metadata-{sp['name']}.csv"
        lines = []
        for rec in db.passes(sid):
            fname = utterance_name(rec["utterance_id"])
            utt = db.utterance(rec["utterance_id"])
            if utt is None:
                continue
            if not (wdir / fname).exists():  # DB bez souboru = nekonzistence
                continue
            lines.append(f"{fname}|{utt['text']}")
        out.write_text("\n".join(lines) + ("\n" if lines else ""),
                       encoding="utf-8")
        total += len(lines)
    return total


def check_wav(path):
    """Header info přes ffprobe → (sample_rate, channels, duration) nebo None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels",
             "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, timeout=30, check=True)
        data = json.loads(out.stdout)
        stream = data["streams"][0]
        return (int(stream["sample_rate"]), int(stream["channels"]),
                float(data["format"]["duration"]))
    except (subprocess.SubprocessError, KeyError, ValueError, IndexError, OSError):
        return None


def speech_gain_db(path, target_db):
    """Stálé zesílení pro jednu nahrávku, aby řeč měla hlasitost target_db. Měří se jen řeč,
    ticho na začátku a konci by výsledek snížilo. Bez dynamické komprese, špičky nepřebudí.
    Mikrofon bez zpracování (iPhone v režimu measurement) dává řeč kolem -45 dB."""
    if np is None or not target_db:
        return 0.0
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            return 0.0
        channels = w.getnchannels()
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    samples = samples[::channels]
    size = int(rate * 0.05)
    if samples.size < size:
        return 0.0
    windows = samples[:samples.size // size * size].reshape(-1, size)
    energy = np.maximum(np.mean(windows ** 2, axis=1), 1e-12)
    levels = 10 * np.log10(energy)
    speech = energy[levels > np.percentile(levels, 10) + 15]
    peak = float(np.max(np.abs(samples)))
    if speech.size == 0 or peak == 0:
        return 0.0
    level = 10 * math.log10(float(np.mean(speech)))
    headroom = PEAK_LIMIT_DB - 20 * math.log10(peak)
    return max(0.0, min(target_db - level, headroom, MAX_GAIN_DB))


def ensure_format(path, sample_rate, highpass_hz=0, gain_db=0.0):
    """Konverze na mono/PCM16/sample_rate (in-place přes tmp + atomic rename).
    highpass_hz > 0 odřízne hučení pozadí pod touto frekvencí (strmě, dvakrát za sebou).
    gain_db je stálé zesílení celé nahrávky."""
    path = Path(path)
    info = check_wav(path)
    if info is None:
        return None
    sr, ch, _ = info
    if not highpass_hz and not gain_db and sr == sample_rate and ch == 1:
        # sjednotit i na PCM16 (whisper/master mohly mít jiný formát)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name", "-of", "json", str(path)],
            capture_output=True, timeout=30, check=True)
        codec = json.loads(probe.stdout)["streams"][0].get("codec_name")
        if codec == "pcm_s16le":
            return info
    tmp = path.with_suffix(".tmp.wav")
    filters = [f"highpass=f={highpass_hz},highpass=f={highpass_hz}"] if highpass_hz else []
    if gain_db:
        filters.append(f"volume={gain_db:.2f}dB")
    af = ["-af", ",".join(filters)] if filters else []
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), *af,
         "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(tmp)],
        check=True, timeout=120)
    tmp.replace(path)
    return check_wav(path)


def build_training_dataset(db, cfg, run_name, speaker_id=None, sample_rate=None):
    """Cumulative dataset vN — metadata + konvertovaná WAV pro piper training.
    Vždy obsahuje VŠECHNY PASS nahrávky daného mluvčího (staré + nové)."""
    sp = db.speaker(speaker_id) if speaker_id is not None else None
    sp_name = sp["name"] if sp else None
    sr = sample_rate or base_sample_rate(cfg)
    run_dir = training_root(cfg) / run_name
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wdir = wav_dir(cfg, sp_name)
    count = 0
    total_dur = 0.0
    for rec in db.passes(speaker_id):
        utt = db.utterance(rec["utterance_id"])
        src = wdir / utterance_name(rec["utterance_id"])
        if src.exists() and utt is not None:
            dst = audio_dir / utterance_name(rec["utterance_id"])
            if not dst.exists():
                shutil.copy2(src, dst)
                # zdroj zůstává surový, hlasitost se srovná až v kopii pro trénink
                gain = speech_gain_db(src, cfg["training"].get("speech_level_db"))
                ensure_format(dst, sr, cfg["training"]["highpass_hz"], gain)
            count += 1
            total_dur += rec["duration"] or 0.0
    # metadata.csv (bez duplicity textu, Piper formát)
    lines = []
    for rec in db.passes(speaker_id):
        utt = db.utterance(rec["utterance_id"])
        if utt is not None and (audio_dir / utterance_name(rec["utterance_id"])).exists():
            lines.append(f"{utterance_name(rec['utterance_id'])}|{utt['text']}")
    (run_dir / "metadata.csv").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {"run_dir": str(run_dir), "sample_rate": sr, "count": count,
            "duration": round(total_dur, 1)}