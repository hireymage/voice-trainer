"""Trénink Piper (piper1-gpl, Lightning DDP) — detached proces, GPU preflight.

Klíčové invarianty:
- Dataset vN = VŠECHNY PASS nahrávky k danému datu (staré + nové), nikdy jen nové.
- Checkpointy append-only (Lightning: top-5 val_mel + last.ckpt per run dir).
- Trénink běží v setsid detached procesu — zavření ssh terminálu ho nezastaví.
- GPU lanes se stopnou jen v tréninkovém procesu (NOPASSWD sudo units).
- Profily (2026-09-26): run patří dvojici (mluvčí × base model); každá dvojice
  má vlastní checkpoint chain — stejná data se dají trénovat z různých bazí.
"""
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
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


def restart_lanes(cfg, run_dir):
    """Znovu spustí služby z gpu_restart_units, které tenhle run zastavil. Načítání modelu
    může trvat minuty, proto se na výsledek nečeká."""
    stopped = _read_json(Path(run_dir) / "stopped_units.json")
    units = [u for u in (stopped if isinstance(stopped, list) else [])
             if u in (cfg.get("gpu_restart_units") or [])]
    for unit in units:
        subprocess.Popen(["sudo", "-n", "systemctl", "start", unit],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    return units


def lane_states(cfg):
    return {u: subprocess.run(["systemctl", "is-active", u], capture_output=True,
                              timeout=15).stdout.decode().strip()
            for u in cfg["gpu_lane_units"]}


def _training_pids(exclude_pid=None, run_dir=None):
    """Procesy tréninku; s run_dir jen toho runu včetně čekání ve frontě. Čte /proc přímo a
    filtruje na python procesy — pgrep -f by matchoval i bash wrappery, které mají
    'piper_fit' jen ve vlastním příkazu (monitoring, ssh -c ...)."""
    import errno
    pids = []
    markers = ["voice_trainer.piper_fit", "piper.train", "voice_trainer.trainer_runner"]
    if run_dir is not None:
        markers.append("voice_trainer.train_queue")
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
        # export ONNX je taky piper.train, ale GPU nezabírá
        if "export_onnx" in cmdline or not any(m in cmdline for m in markers):
            continue
        if run_dir is None or f"{run_dir}/" in cmdline:
            pids.append(pid)
    return pids


def training_running(cfg, exclude_pid=None):
    """Běží nějaký piper fit / runner?"""
    return bool(_training_pids(exclude_pid))


def run_alive(run):
    return bool(_training_pids(run_dir=run["dataset_dir"]))


def stop_run(db, cfg, run, wait_s=30):
    """Zastaví daný run. Procesy se hledají podle složky runu, takže to nesáhne na jiný
    trénink. Uložený last.ckpt zůstává a další run na něj naváže."""
    run_dir = run["dataset_dir"]

    def send(sig):
        pids = _training_pids(run_dir=run_dir)
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        return pids

    send(signal.SIGTERM)
    deadline = time.time() + wait_s
    while _training_pids(run_dir=run_dir) and time.time() < deadline:
        time.sleep(1)
    if send(signal.SIGKILL):
        time.sleep(2)  # GPU se uvolní až po skutečném konci procesů
    db.update_run(run["id"], "stopped", notes="zastaveno")
    status_path = Path(run_dir) / "status.json"
    status = _read_json(status_path)
    status.update(status="stopped", updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    status_path.write_text(json.dumps(status, indent=2))
    return restart_lanes(cfg, run_dir)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _fit_log_last(run_dir, pattern):
    """Poslední výskyt čísla v logu Lightningu; log je dlouhý, čte se jen konec."""
    try:
        with open(Path(run_dir) / "fit.log", "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            found = re.findall(pattern, f.read())
    except OSError:
        return None
    return found[-1] if found else None


def _fit_epoch(run_dir):
    found = _fit_log_last(run_dir, rb"Epoch (\d+)")
    return int(found) if found is not None else None


def run_quality(run):
    """(poslední, nejlepší) val_mos runu: odhad kvality 1–5, vyšší je lepší. Bez prediktoru MOS None."""
    run_dir = Path(run["dataset_dir"])
    latest = _fit_log_last(run_dir, rb"val_mos=(\d+\.\d+)")
    saved = [float(m.group(1)) for p in run_dir.rglob("epoch=*-val_mos=*.ckpt")
             if (m := re.search(r"val_mos=(\d+\.\d+)", p.name))]
    return float(latest) if latest is not None else None, max(saved, default=None)


def progress(db, cfg, speaker_id):
    """Stav posledního runu hlasu pro klienta: state, procenta a odhad zbývajícího času."""
    passes, _ = db.stats(speaker_id)
    out = {"state": "idle", "run": None, "percent": None, "epoch_done": None,
           "epoch_total": None, "eta_s": None, "samples": None, "new_recordings": 0,
           "passes": passes, "error": None, "ended_at": None}
    run = db.latest_run(speaker_id=speaker_id)
    if run is None:
        return out
    run_dir = Path(run["dataset_dir"])
    status = _read_json(run_dir / "status.json")
    state = run["status"]
    alive = run_alive(run)
    if state == "running":
        state = (status.get("status") or "preparing") if alive else "failed"
    if state == "failed":
        # runner po pádu řádek v DB nestihl přepsat
        out["error"] = status.get("reason") or run["notes"] or "trénink se nečekaně ukončil"
    samples = (status.get("dataset") or {}).get("count")
    out.update(state=state, run=run["name"], samples=samples, ended_at=run["ended_at"],
               new_recordings=max(0, passes - samples) if samples is not None else 0)

    # Lightning čísluje epochy dál od základu: první je base+1, poslední base+max_epochs-1
    max_epochs = int(_read_json(run_dir / "train_request.json").get("max_epochs") or 0)
    if max_epochs < 2:
        return out
    meta_path = run_dir / "base_meta.json"
    base_epoch = int(_read_json(meta_path).get("epoch") or 0)
    total = max_epochs - 1
    epoch = _fit_epoch(run_dir)
    done = total if state == "done" else \
        min(total, max(0, epoch - base_epoch - 1)) if epoch is not None else 0
    out.update(epoch_done=done, epoch_total=total, percent=round(done / total * 100, 1))
    if state == "training" and done >= 3 and meta_path.exists():
        # base_meta.json vzniká těsně před fitem, měří se tedy i příprava keše
        elapsed = time.time() - meta_path.stat().st_mtime
        out["eta_s"] = int(elapsed / done * (total - done))
    return out


def busy_speaker(db, speaker_id):
    """Jméno jiného hlasu, který právě trénuje (GPU jsou jen jedny)."""
    for run in db.all("SELECT r.*, s.name sp_name FROM runs r JOIN speakers s "
                      "ON s.id = r.speaker_id WHERE r.status IN ('running','queued') "
                      "AND r.speaker_id <> ?", (speaker_id,)):
        if run_alive(run):
            return run["sp_name"]
    return None


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
    sql = ("SELECT * FROM runs WHERE status IN ('done','running','stopped')")
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


def _run_piper(cmd, timeout):
    # bez GPU: export i zkouška musí jít i při běžícím tréninku
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    if proc.returncode != 0:
        lines = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"{cmd[2]} selhal")


def export_onnx(cfg, checkpoint, output_file):
    """piper.train.export_onnx → .onnx + .onnx.json (config z run dir)."""
    t = cfg["training"]
    py = str(Path(t["piper_venv"]) / "bin" / "python")
    _run_piper([py, "-m", "piper.train.export_onnx", "--checkpoint", checkpoint,
                "--output-file", output_file], 1800)
    return output_file


def trained_checkpoint(db, speaker_id):
    """Poslední uložený stav hlasu a jeho run. base.ckpt se nepočítá, ten by zněl cizím hlasem."""
    for run in db.all("SELECT * FROM runs WHERE speaker_id=? AND status IN "
                      "('done','running','stopped') ORDER BY id DESC", (speaker_id,)):
        found = sorted(Path(run["dataset_dir"]).rglob("last.ckpt"),
                       key=lambda p: p.stat().st_mtime)
        if found:
            return found[-1], run
    return None, None


def _model_paths(cfg, speaker_name):
    outdir = Path(cfg["data_root"]) / "export" / "app"
    onnx = outdir / f"{speaker_name}.onnx"
    return outdir, onnx, Path(f"{onnx}.json"), outdir / f"{speaker_name}.source"


def app_model(db, cfg, speaker_id):
    """Model z posledního stavu hlasu pro zkoušku a nasazení: (onnx, config, run), nebo None.
    Stejný stav se znovu neexportuje a exporty z CLI to nepřepisuje."""
    ckpt, run = trained_checkpoint(db, speaker_id)
    if ckpt is None:
        return None
    outdir, onnx, config, source = _model_paths(cfg, db.speaker(speaker_id)["name"])
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=outdir) as tmp:
        # během tréninku se last.ckpt přepisuje; kopie musí být z jednoho stavu
        copy = Path(tmp) / "model.ckpt"
        for _ in range(5):
            stamp = ckpt.stat().st_mtime_ns
            key = f"{ckpt}:{stamp}"
            if onnx.exists() and config.exists() and source.exists() \
                    and source.read_text() == key:
                return onnx, config, run
            shutil.copy2(ckpt, copy)
            if ckpt.stat().st_mtime_ns == stamp:
                break
            time.sleep(2)
        else:
            raise RuntimeError("checkpoint se pořád přepisuje, zkus to za chvíli")
        new = Path(tmp) / "model.onnx"
        export_onnx(cfg, str(copy), str(new))
        new.replace(onnx)
        shutil.copy2(Path(run["dataset_dir"]) / "config.json", config)
        source.write_text(key)
    return onnx, config, run


def published_at(cfg, speaker_name):
    """Kdy se model hlasu naposledy nasadil přes publish_command."""
    outdir, _, _, _ = _model_paths(cfg, speaker_name)
    try:
        stamp = (outdir / f"{speaker_name}.published").stat().st_mtime
    except OSError:
        return None
    return datetime.datetime.fromtimestamp(stamp).strftime("%Y-%m-%dT%H:%M:%S")


def publish_model(db, cfg, speaker_id):
    """Spustí publish_command z configu; kam se model nasazuje, ví jen ten příkaz.
    None, když hlas ještě nemá co nasadit."""
    model = app_model(db, cfg, speaker_id)
    if model is None:
        return None
    onnx, config, run = model
    sp = db.speaker(speaker_id)
    proc = subprocess.run(
        cfg["publish_command"], shell=True, capture_output=True, timeout=600, cwd=cfg["repo_dir"],
        env={**os.environ, "VT_MODEL": str(onnx), "VT_MODEL_CONFIG": str(config),
             "VT_SPEAKER": sp["name"], "VT_LANGUAGE": sp["language"], "VT_RUN": run["name"]})
    if proc.returncode != 0:
        lines = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"publish_command skončil s {proc.returncode}")
    (onnx.parent / f"{sp['name']}.published").touch()
    return run["name"]


def test_inference(cfg, model_path, text, out_wav):
    """Piper runtime inference → WAV (test vytvořeného modelu)."""
    t = cfg["training"]
    py = str(Path(t["piper_venv"]) / "bin" / "python")
    _run_piper([py, "-m", "piper", "-m", str(model_path), "-f", str(out_wav), "--", text], 300)
    return out_wav