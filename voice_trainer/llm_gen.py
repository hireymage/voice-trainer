"""Pozadí generování vět — detached proces spouštěný cmd_next_sentence.

Účel: hranice dávky (pool vyčerpaný → LLM generuje 2+ min) nesmí zastavit
nahrávání. cmd_next_sentence spustí tenhle modul, jakmile zbývá ≤3 vět;
dokud uživatel dořikná dávku, další je už v DB.

Konkurence: flock na <data_root>/.llm-gen.lock — kdo ho nedostane non-blocking
přístupem, jiný generátor už běží a modul hned končí."""
import fcntl
import sys
from pathlib import Path

from voice_trainer import db as dbmod
from voice_trainer import llm as llm_mod
from voice_trainer.config import load


def main():
    cfg = load(sys.argv[1] if len(sys.argv) > 1 else None)
    lock_path = Path(cfg["data_root"]) / ".llm-gen.lock"
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return  # generátor už běží — druhý spád neběží
    db = dbmod.Db(Path(cfg["data_root"]) / "voice-trainer.db")
    try:
        for s in llm_mod.generate(cfg):
            db.add_utterance(s, "llm")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()