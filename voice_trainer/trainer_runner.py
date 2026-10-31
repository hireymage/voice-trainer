"""Detached tréninkový runner — spouští ho training.start_training (setsid).

Dělá celý životní cyklus runu: dataset build → GPU preflight (stop lanes)
→ piper fit (Lightning DDP) → status done/failed. Vše loguje do
training.log, stav do status.json (čte ho `voice-trainer status`).
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_trainer import dataset as ds       # noqa: E402
from voice_trainer import db as dbmod          # noqa: E402
from voice_trainer import training as tr       # noqa: E402
from voice_trainer.config import load          # noqa: E402


def log(run_dir, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(Path(run_dir) / "training.log", "a") as f:
        f.write(line + "\n")


def write_status(run_dir, data):
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(Path(run_dir) / "status.json", "w") as f:
        json.dump(data, f, indent=2)


def main():
    req = json.load(open(sys.argv[1]))
    run_dir = req["run_dir"]
    cfg = load()
    db = dbmod.Db(Path(cfg["data_root"]) / "voice-trainer.db")

    # SIGTERM → graceful: když už fit běží, ukončit i jeho procesní skupinu
    # (jinak fit zůstane osiřelý s PPID 1 a bude točit GPU — viz run_003)
    state = {}
    def _on_term(signum, _frame):
        proc = state.get("fit_proc")
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        sys.exit(143)
    signal.signal(signal.SIGTERM, _on_term)

    write_status(run_dir, {"status": "preparing", "run": req["run_name"]})
    log(run_dir, f"run {req['run_name']} — mluvčí={req.get('speaker_name') or '?'}"
                 f" base={req.get('base_name') or '?'} ({req['base_checkpoint']})")

    # 1) cumulative dataset (VŠECHNY PASS nahrávky daného mluvčího)
    d = ds.build_training_dataset(db, cfg, req["run_name"],
                                  speaker_id=req.get("speaker_id"),
                                  sample_rate=req.get("sample_rate"))
    log(run_dir, f"dataset vN: {d['count']} samples, {d['duration']}s, sr={d['sample_rate']}")
    if d["count"] < 4:
        db.update_run(req["run_id"], "failed", notes="méně než 4 PASS nahrávky")
        write_status(run_dir, {"status": "failed", "reason": "málo dat"})
        return

    # 2) GPU preflight — stop lanes, ověřit volnou VRAM
    gpus = tr.gpu_status()
    log(run_dir, "GPU před stopnutím lanes: " + json.dumps(gpus))
    stopped = tr.stop_lanes(cfg)
    log(run_dir, f"stopnuté lanes: {stopped or 'žádné'}")
    time.sleep(3)
    gpus = tr.gpu_status()
    log(run_dir, "GPU po stopnutí lanes: " + json.dumps(gpus))
    write_status(run_dir, {"status": "preparing", "gpus": gpus,
                           "dataset": d, "run": req["run_name"]})

    # 3) fit — base checkpoint nejdřív sanitovat (starý formát hyper_parameters)
    # clean_ckpt importuje piper.train → musí běžet v piper venv (subprocess)
    base_for_fit = req["base_checkpoint"]
    base_epoch = 0
    if base_for_fit and Path(base_for_fit).is_file():
        clean_path = Path(run_dir) / "base.ckpt"
        meta_path = Path(run_dir) / "base_meta.json"
        if not clean_path.exists():
            log(run_dir, f"sanitace base checkpointu → {clean_path}")
            py_piper = str(Path(cfg["training"]["piper_venv"]) / "bin" / "python")
            r = subprocess.run(
                [py_piper, "-m", "voice_trainer.clean_ckpt",
                 base_for_fit, str(clean_path)],
                cwd=cfg["repo_dir"], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("clean_ckpt selhal: "
                                   + (r.stderr or r.stdout)[-500:])
            log(run_dir, r.stdout.strip())
            m = re.search(r"epoch: (\d+)", r.stdout)
            base_meta = {"epoch": int(m.group(1)) if m else 0}
            json.dump(base_meta, open(meta_path, "w"))
        elif meta_path.exists():
            base_meta = json.load(open(meta_path))
            base_meta = base_meta if isinstance(base_meta, dict) else {}
        else:
            base_meta = {}
        base_epoch = int(base_meta.get("epoch") or 0)
        base_for_fit = str(clean_path)
    # Lightning chce ABSOLUTNÍ max_epochs (base epoch + nové epochy)
    max_epochs_total = base_epoch + int(req["max_epochs"])
    cmd = tr.build_fit_cmd(cfg, run_dir, req["sample_rate"],
                           base_for_fit, max_epochs_total,
                           voice_name=req.get("speaker_name") or "voice-trainer",
                           espeak_voice=req.get("espeak_voice"))
    log(run_dir, "fit: " + " ".join(cmd))
    env = {"NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1",
           "NCCL_DEBUG": "WARN", "PATH": "/usr/local/sbin:/usr/local/bin:"
           "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": str(Path.home())}
    write_status(run_dir, {"status": "training", "dataset": d,
                           "command": " ".join(cmd), "run": req["run_name"]})
    started = time.time()
    with open(Path(run_dir) / "fit.log", "w") as flog:
        proc = subprocess.Popen(
            cmd, stdout=flog, stderr=subprocess.STDOUT, env=env,
            start_new_session=True)
        state["fit_proc"] = proc
        rc = proc.wait()
        if rc != 0:
            # DDP rank mohl spadnout, zatímco sourozenci žijí dál (NCCL
            # bariéra) — ukončit celou procesní skupinu graceful
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    mins = (time.time() - started) / 60
    log(run_dir, f"fit ukončen rc={rc} po {mins:.1f} min")
    status = "done" if rc == 0 else "failed"
    db.update_run(req["run_id"], status, notes=f"rc={rc}, {mins:.1f} min")
    write_status(run_dir, {"status": status, "dataset": d,
                           "return_code": rc, "minutes": round(mins, 1),
                           "run": req["run_name"]})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # výpadek GPU/disku nesmí zničit metadata — run označit failed a logovat
        try:
            log(json.load(open(sys.argv[1]))["run_dir"],
                "FATAL: " + traceback.format_exc())
        except Exception:
            traceback.print_exc()
        sys.exit(1)