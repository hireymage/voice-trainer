"""Wrapper pro `python piper_fit.py fit` — gotchy piper1-gpl na torch 2.x + DDP.

1. weights_only: base checkpointy (rhasspy/piper-checkpoints) obsahují
   pathlib.PosixPath, které LightningCLI `_parse_ckpt_path` (torch.load
   weights_only=True) odmítne. Zdroj je trusted (oficiální Piper
   checkpointy), takže stačí allowlist typu.
2. DDP prepare_data: Lightning volá `prepare_data` jen na ranku 0, ale
   VitsDataModule nastavuje `self.piper_config` uvnitř → rank1 spadne na
   `assert piper_config is not None`. Rank0 config.json zapsal, takže v
   setupu ho ostatní ranky prostě do-loadí.
"""
import json
import pathlib

import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])

from piper.config import PiperConfig           # noqa: E402
from piper.train.vits.dataset import VitsDataModule  # noqa: E402

_orig_setup = VitsDataModule.setup


def _setup_all_ranks(self, stage):
    if getattr(self, "piper_config", None) is None and self.config_path.is_file():
        with open(self.config_path, encoding="utf-8") as f:
            self.piper_config = PiperConfig.from_dict(json.load(f))
    return _orig_setup(self, stage)


VitsDataModule.setup = _setup_all_ranks

from piper.train.__main__ import main          # noqa: E402

main()