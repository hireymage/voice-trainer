"""Sanitace base checkpointu pro finetune (starý Piper formát hyper_parameters).

Checkpointy z rhasspy/piper-checkpoints vznikly starou verzí piperu — jejich
`hyper_parameters` jsou plochý dict se jmény, která aktuální piper1-gpl
nezná (sample_bytes, quality, gpus, …). LightningCLI při `--ckpt_path` tyto
hparams merguje do `model` skupiny S PŘEDNOSTÍ před CLI a při neznámém klíči
celý parse padá. Sanitace proto ponechá jen klíče, které aktuální VitsModel
skutečně akceptuje, minus `batch_size` (staré 100 by přepsalo náš config).
Trainer/data nastavení řídí výhradně náš fit příkaz.
"""
import inspect
import pathlib
import sys
from pathlib import Path

import torch


def clean(base_ckpt, out_ckpt, drop=("batch_size",)):
    """Načte base checkpoint, vyfiltrí hyper_parameters a uloží do out_ckpt.
    Vrací (kept_keys, dropped_keys, base_epoch)."""
    # old Piper checkpointy obsahují PosixPath v hparams (trusted zdroj)
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    ckpt = torch.load(base_ckpt, weights_only=True, map_location="cpu")
    base_epoch = int(ckpt.get("epoch") or 0)

    # introspekce aktuálního VitsModel (spouštěno v piper venv/piper repo)
    from piper.train.vits.lightning import VitsModel

    model_keys = set(inspect.signature(VitsModel.__init__).parameters)
    hp = dict(ckpt.get("hyper_parameters") or {})
    hp.pop("_instantiator", None)
    kept, dropped = {}, []
    for k, v in hp.items():
        if k in model_keys and k not in drop:
            kept[k] = v
        else:
            dropped.append(k)
    ckpt["hyper_parameters"] = kept
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_ckpt) + ".tmp"
    torch.save(ckpt, tmp)
    Path(tmp).replace(out_ckpt)  # atomic
    return sorted(kept), dropped, base_epoch


if __name__ == "__main__":
    # jen pro ruční otestování; runner volá clean() přes subprocess a parsuje
    # řádek "epoch: N" pro výpočet absolutního --trainer.max_epochs
    kept, dropped, base_epoch = clean(sys.argv[1], sys.argv[2])
    print("epoch:", base_epoch)
    print("kept:", len(kept))