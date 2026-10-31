"""Konfigurace voice-traineru — config.yaml, žádné IP v kódu."""
import os
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parent.parent

_DEFAULTS = {
    "data_root": "/mnt/models/voice-trainer",
    "repo_dir": str(REPO_DIR),
    "stt_url": "http://localhost:8082/inference",
    "stt_language": "cs",
    "stt_timeout": 300,
    "stt_retries": 2,
    "api_host": "127.0.0.1",
    "api_port": 8097,
    "llm_url": "http://localhost:11434",
    "llm_model": "glm-5.3-flash:cloud",
    "llm_batch_size": 10,
    "llm_timeout": 120,
    "recording": {
        "sample_rate": 48000,
        "stt_sample_rate": 16000,
        "max_seconds": 120,
        "similarity_threshold": 0.90,
    },
    "training": {
        "fallback_sample_rate": 22050,
        "highpass_hz": 80,
        "espeak_voice": "cs",
        "devices": 2,
        "strategy": "ddp",
        "batch_size": 8,
        "precision": "32-true",
        "max_epochs": 500,
        "val_ratio": 0.02,
        "base_checkpoint": "",
        "base_config": "",
        "piper_repo": "",
        "piper_venv": "",
    },
    "gpu_lane_units": [
        "llama-server.service",
        "llama-server-b.service",
        "llama-server-bonsai0.service",
        "llama-server-bonsai1.service",
    ],
    "quality": {
        "min_duration": 1.0,
        "max_duration": 30.0,
        "clip_peak": 0.99,
        "clip_count": 20,
        "quiet_rms_db": -35.0,
        "loud_rms_db": -10.0,
    },
}


def _merge(base, extra):
    out = dict(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path=None):
    """Načte config.yaml (default: repo_dir/config.yaml). Vrací dict s defaults."""
    cfg_path = Path(path or os.environ.get("VOICE_TRAINER_CONFIG")
                    or (REPO_DIR / "config.yaml"))
    data = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
    return _merge(_DEFAULTS, data)