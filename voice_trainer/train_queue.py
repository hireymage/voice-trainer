"""Čekací fronta tréninku — auto-trigger, když už fit běží.

start_training(defer=True) spustí tenhle detached waiter: polluje GPU,
a jakmile žádný piper fit neběží, pustí trainer_runner pro svůj run.
Pozn.: base_checkpoint v requestu se vyhodnocuje až v momentě startu —
pokud se mezitím dokončil další run TÉŽE DVOJICE (mluvčí × base),
převezme jeho last.ckpt (kumulace per dvojice).
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_trainer import training as tr  # noqa: E402


def main():
    req = json.load(open(sys.argv[1]))
    run_dir = Path(req["run_dir"])
    lock_path = run_dir.parent / ".train.lock"

    # 1) získej frontovní zámek (jen JEDEN čekač smí pustit runner —
    #    jinak se dva probudí ve stejnou chvíli na volné GPU a pustí se
    #    oba, viz run_005 + run_006). Zámek držíme přes celý běh runneru;
    #    uvolní se sám smrtí procesu (i po SIGKILL, jádro to uklidí).
    deadline = time.time() + 24 * 3600
    lock_f = open(lock_path, "w")
    while time.time() < deadline:
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            time.sleep(20)
    else:
        Path(run_dir / "training.log").open("a").write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FATAL: fronta timeout "
            f"(24 h) — GPU/lock nikdy nebyl uvolněn\n")
        sys.exit(1)

    # 2) pojistka: fit může běžet i bez zámku (ruční start) — počkej na něj
    while tr.training_running(req):
        if time.time() > deadline:
            Path(run_dir / "training.log").open("a").write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FATAL: fronta "
                f"timeout (24 h) — GPU nikdy nebyla uvolněna\n")
            sys.exit(1)
        time.sleep(30)

    # 3) re-resolve base na nejnovější dokončený checkpoint TÉŽE DVOJICE
    #    (mluvčí × base) — řetězce jsou per dvojici nezávislé
    from voice_trainer import db as dbmod
    from voice_trainer.config import load
    cfg = load()
    db = dbmod.Db(Path(cfg["data_root"]) / "voice-trainer.db")
    latest = tr.find_checkpoint(db, cfg, "last",
                                speaker_id=req.get("speaker_id"),
                                base_name=req.get("base_name"))
    if latest:
        req["base_checkpoint"] = latest
        payload_path = run_dir / "train_request.json"
        json.dump(req, open(payload_path, "w"), indent=2)

    # 4) pustit runner synchronně (dědí detached session z launcheru)
    r = subprocess.run(
        [sys.executable, "-m", "voice_trainer.trainer_runner", sys.argv[1]],
        cwd=req["repo_dir"])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()