"""Trénink Piper (piper1-gpl, Lightning DDP) — detached proces, GPU preflight.

Klíčové invarianty:
- Dataset vN = VŠECHNY PASS nahrávky k danému datu (staré + nové), nikdy jen nové.
- Checkpointy append-only (Lightning: top-5 val_mel + last.ckpt per run dir).
- Trénink běží v setsid detached procesu — zavření ssh terminálu ho nezastaví.
- GPU lanes se stopnou jen v tréninkovém procesu (NOPASSWD sudo units).
- Profily (2026-09-26): run patří dvojici (mluvčí × base model); každá dvojice
  má vlastní checkpoint chain — stejná data se dají trénovat z různých bazí.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from voice_trainer import dataset as ds
from voice_trainer import db as dbmod
from voice_trainer import languages


def gpu_status():
    """(index, name, total MiB, free MiB) přes nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=15, check=True).stdout.decode()
        gpus = []
        for line in out.strip().splitlines():
            idx, name, total, free = [p.strip() for p in line.split(",")]
            gpus.append({"index": int(idx), "name": str(name),
                         "total_mb": int(total), "free_mb": int(free)})
        return gpus
    except (subprocess.SubprocessError, ValueError, OSError) as e:
        raise RuntimeError(f"nvidia-smi selhal: {e}")


def stop_lanes(cfg):
    """Graceful stop GPU lane units (NOPASSWD sudo).
    Vrací seznam stopnutých units."""
    stopped = []
    for unit in cfg["gpu_lane_units"]:
        state = subprocess.run(["systemctl", "is-active", unit],
                               capture_output=True, timeout=15).stdout.decode().strip()
        if state != "active":
            continue
        subprocess.run(["sudo", "-n", "systemctl", "stop", unit],
                       capture_output=True, timeout=60, check=False)
        stopped.append(unit)
    return stopped


def lane_states(cfg):
    return {u: subprocess.run(["systemctl", "is-active", u], capture_output=True,
                              timeout=15).stdout.decode().strip()
            for u in cfg["gpu_lane_units"]}


def training_running(cfg, exclude_pid=None):
    """Běží nějaký piper fit / runner? Čte /proc přímo a filtruje na
    python procesy — pgrep -f by matchoval i bash wrappery, které mají
    'piper_fit' jen ve vlastním příkazu (monitoring, ssh -c ...)."""
    import errno
    for pdir in Path("/proc").iterdir():
        if not pdir.name.isdigit():
            continue
        pid = int(pdir.name)
        if exclude_pid is not None and pid == exclude_pid:
            continue
        try:
            comm = (pdir / "comm").read_text().strip()
            if "python" not in comm:
                continue
            cmdline = (pdir / "cmdline").read_bytes() \
                .decode("utf-8", "replace")
        except OSError as e:
            if e.errno in (errno.ENOENT, errno.ESRCH, errno.EACCES):
                continue
            raise
        if ("voice_trainer.piper_fit" in cmdline
                or "piper.train" in cmdline
                or "voice_trainer.trainer_runner" in cmdline):
            return True
    return False


def bases(cfg, language=None):
    """Dostupné base modely: dict label → {ckpt, config, language}, případně jen pro jeden jazyk.
    Config definuje training.bases (mapa label → cesty a language, výchozí cs); pokud chybí,
    fallback z legacy klíčů base_checkpoint/base_config pod label 'jirka'."""
    t = cfg["training"]
    out = {}
    for label, b in (t.get("bases") or {}).items():
        if isinstance(b, dict):
            out[label] = {"ckpt": b.get("checkpoint") or "",
                          "config": b.get("config") or "",
                          "language": b.get("language") or languages.DEFAULT_LANGUAGE}
    if not out and t.get("base_checkpoint"):
        out["jirka"] = {"ckpt": t["base_checkpoint"],
                        "config": t.get("base_config") or "",
                        "language": languages.DEFAULT_LANGUAGE}
    if language is not None:
        out = {label: b for label, b in out.items() if b["language"] == language}
    return out


def resolve_base(cfg, base_name=None):
    """label → (ckpt, config) nebo (None, None) při neznámém labelu."""
    bs = bases(cfg)
    if not bs:
        return None, None
    if base_name is None:
        base_name = next(iter(bs))
    b = bs.get(base_name)
    return (b["ckpt"], b["config"]) if b else (None, None)


def build_fit_cmd(cfg, run_dir, sample_rate, base_ckpt, max_epochs,
                  voice_name="voice-trainer", espeak_voice=None):
    """LightningCLI fit příkaz (piper venv) — argumenty dle TRAINING.md."""
    t = cfg["training"]
    py = str(Path(t["piper_venv"]) / "bin" / "python")
    # keš Piperu je podle pořadí řádku a textu věty, ne podle zvuku ani mluvčího. Sdílená by
    # po novém nahrání smazané věty vrátila starý zvuk, nový filtr by se neprojevil a mluvčí
    # by si mohli míchat nahrávky; proto má každý běh vlastní (příprava trvá jen minuty)
    # wrapper místo -m piper.train: allowlist pathlib.PosixPath pro torch.load
    # (LightningCLI weights_only=True jinak odmítne base checkpointy z rhasspy)
    # runner spouští fit s cwd=repo_dir, takže -m voice_trainer.* je vidět
    cmd = [
        py, "-m", "voice_trainer.piper_fit", "fit",
        "--data.voice_name", str(voice_name),
        "--data.csv_path", str(Path(run_dir) / "metadata.csv"),
        "--data.audio_dir", str(Path(run_dir) / "audio"),
        "--data.cache_dir", str(Path(run_dir) / "cache"),
        "--data.config_path", str(Path(run_dir) / "config.json"),
        "--data.espeak_voice", str(espeak_voice or t["espeak_voice"]),
        "--data.batch_size", str(t["batch_size"]),
        "--data.validation_split", str(t["val_ratio"]),
        "--data.num_test_examples", "2",
        "--model.sample_rate", str(sample_rate),
        "--trainer.accelerator", "gpu",
        "--trainer.devices", str(t["devices"]),
        "--trainer.strategy", str(t["strategy"]),
        "--trainer.precision", str(t["precision"]),
        "--trainer.max_epochs", str(max_epochs),
        "--trainer.default_root_dir", str(run_dir),
        "--data.num_workers", "2",
    ]
    if base_ckpt and Path(base_ckpt).exists():
        cmd += ["--ckpt_path", str(base_ckpt)]
    return cmd


def start_training(db, cfg, base_ckpt=None, max_epochs=None, run_name=None,
                   defer=False, speaker_id=None, base_name=None):
    """Naplánuje trénink: vytvoří run a spustí detached runner (setsid).
    defer=True → run se jen zařadí (status queued); waiter spustí fit,
    jakmile se uvolní GPU (auto-trigger při běžícím tréninku).
    base = dvojice (mluvčí × base label); bez base_ckpt se řetězí kumulativně:
    checkpoint posledního runu téže dvojice → jinak default base daného labelu.
    speaker_id None → jediný mluvčí datasetu (při více → chyba)."""
    t = cfg["training"]
    if speaker_id is None:
        sps = db.speakers()
        if not sps:
            raise SystemExit("dataset nemá žádné nahrávky — nejdřív nahrávej")
        if len(sps) > 1:
            raise SystemExit("více mluvčích — zadej --speaker (např. "
                             + ", ".join(s["name"] for s in sps) + ")")
        speaker_id = sps[0]["id"]
    sp = db.speaker(speaker_id)
    if sp is None:
        raise SystemExit("neznámý mluvčí")
    language = sp["language"]
    # základ v jiném jazyce by hlas naučil špatnou výslovnost
    allowed = bases(cfg, language)
    if not base_ckpt and base_name is None and bases(cfg) and not allowed:
        raise SystemExit(f"pro jazyk {language} není základní model; "
                         "přidej ho do training.bases v config.yaml")
    if base_name is not None and bases(cfg).get(base_name) and base_name not in allowed:
        raise SystemExit(f"base model {base_name} není pro jazyk {language}")
    if base_name is None and not base_ckpt:
        base_name = next(iter(allowed)) if allowed else None
    # bez labelu (vlastní --checkpoint) se česká větev dál řetězí pod původním jménem
    base_label = base_name or ("jirka" if language == languages.DEFAULT_LANGUAGE else "custom")
    ckpt_label, base_config = resolve_base(cfg, base_name)
    if base_name is not None and ckpt_label is None:
        raise SystemExit(f"neznámý base model: {base_name} (config training.bases)")
    passes, dur = db.stats(speaker_id)
    if passes == 0:
        raise SystemExit(
            f"mluvčí {sp['name']} nemá žádné PASS nahrávky — nejdřív nahrávej")
    base = base_ckpt or find_checkpoint(db, cfg, "last", speaker_id, base_name) \
        or ckpt_label
    sr = ds.base_sample_rate(cfg, base_config)
    name = run_name or db.new_run_name()
    run_dir = ds.training_root(cfg) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    run = db.create_run(name, str(run_dir), base, speaker_id=speaker_id,
                        base_name=base_label)
    if defer:
        db.update_run(run["id"], "queued", notes="čeká na uvolnění GPU")
    payload = {
        "run_id": run["id"], "run_name": name, "run_dir": str(run_dir),
        "base_checkpoint": base, "base_config": base_config or "",
        "base_name": base_label,
        "speaker_id": speaker_id, "speaker_name": sp["name"],
        "sample_rate": sr,
        "max_epochs": max_epochs or int(t["max_epochs"]),
        "repo_dir": cfg["repo_dir"], "data_root": cfg["data_root"],
        "gpu_lane_units": cfg["gpu_lane_units"],
        "piper_venv": t["piper_venv"], "piper_repo": t["piper_repo"],
        "batch_size": t["batch_size"], "devices": t["devices"],
        "strategy": t["strategy"], "precision": t["precision"],
        "espeak_voice": languages.get(cfg, language)["espeak_voice"],
        "val_ratio": t["val_ratio"],
        "recording": cfg["recording"],
        "quality": cfg["quality"],
    }
    json_path = run_dir / "train_request.json"
    json.dump(payload, open(json_path, "w"), indent=2)
    log = open(run_dir / "launcher.log", "w")
    entry = "train_queue" if defer else "trainer_runner"
    subprocess.Popen(
        [sys.executable, "-m", "voice_trainer." + entry, str(json_path)],
        stdout=log, stderr=subprocess.STDOUT,
        cwd=cfg["repo_dir"], start_new_session=True)
    return {"run": name, "run_dir": str(run_dir), "samples": passes,
            "duration_s": round(dur, 1), "sample_rate": sr, "base": base,
            "base_name": base_label, "speaker": sp["name"],
            "queued": defer}


def find_checkpoint(db, cfg, which="last", speaker_id=None, base_name=None):
    """Checkpoint posledního runu daného mluvčího a base modelu ve stavu
    done/running — novější failed/queued řádky se přeskočí (jinak jeden
    zastavený run zablokuje řetězení). Dvojice (speaker × base) = nezávislé
    řetězce; NULL sloupce (legacy) matchnou jen NULL query — po migraci
    jsou ale všechny runy backfillnuté."""
    sql = ("SELECT * FROM runs WHERE status IN ('done','running')")
    args = []
    if speaker_id is not None:
        sql += " AND speaker_id = ?"
        args.append(speaker_id)
    if base_name is not None:
        sql += " AND base_name = ?"
        args.append(base_name)
    sql += " ORDER BY id DESC LIMIT 1"
    run = db.one(sql, args)
    if run is None:
        return None
    cdir = Path(run["dataset_dir"])
    if which == "last":
        # Lightning last.ckpt je buď v default_root_dir, nebo v lightning_logs.
        # U běžícího runu last.ckpt ještě není — base.ckpt (jeho vyčištěný
        # základ) je pak nejlepší dostupný bod pro řetězení.
        for p in (cdir / "checkpoints" / "last.ckpt",
                  cdir / "last.ckpt",
                  cdir / "lightning_logs" / "version_0" / "checkpoints" / "last.ckpt",
                  cdir / "base.ckpt"):
            if p.exists():
                return str(p)
        found = sorted(cdir.rglob("last.ckpt"))
        return str(found[-1]) if found else None
    best = sorted(cdir.rglob("epoch=*.ckpt"))
    return str(best[-1]) if best else None


def export_onnx(cfg, checkpoint, output_file):
    """piper.train.export_onnx → .onnx + .onnx.json (config z run dir)."""
    t = cfg["training"]
    py = str(Path(t["piper_venv"]) / "bin" / "python")
    cmd = [py, "-m", "piper.train.export_onnx", "--checkpoint", checkpoint,
           "--output-file", output_file]
    subprocess.run(cmd, check=True, timeout=1800)
    return output_file


def test_inference(cfg, model_path, text, out_wav):
    """Piper runtime inference → WAV (test vytvořeného modelu)."""
    t = cfg["training"]
    py = str(Path(t["piper_venv"]) / "bin" / "python")
    cmd = [py, "-m", "piper", "-m", str(model_path), "-f", str(out_wav), "--", text]
    subprocess.run(cmd, check=True, timeout=300)
    return out_wav