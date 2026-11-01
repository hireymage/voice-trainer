# voice-trainer

Serverová část pro trénink vlastního [Piper](https://github.com/OHF-Voice/piper1-gpl)
TTS hlasu v češtině, angličtině nebo dalším jazyce: dataset, kontrola nahrávek přes STT, trénink na GPU a export do ONNX.
Klient nahrává, server počítá.

```
TEXT (LLM / import) → NAHRÁVKA (klient) → STT KONTROLA (whisper)
→ DATASET (SQLite + WAV) → PIPER TRAINING (DDP, detached)
→ CHECKPOINT (append-only) → ONNX → PIPER INFERENCE
```

Nahrávací klienti nejsou součástí tohoto repozitáře. Se serverem mluví přes HTTP API
(`voice_trainer/api.py`) nebo přes JSON výstup CLI.

## Komponenty

| Co | Poznámka |
|---|---|
| `voice-trainer` (CLI) | dataset, trénink, export |
| `voice-trainer api` | HTTP API pro klienty, Bearer token |
| LLM (generování vět) | ollama, model podle `llm_model` |
| STT | whisper.cpp server, `stt_url` |
| Data + venvy | `data_root`, stačí HDD |
| Piper training | piper1-gpl + vlastní venv, Lightning DDP |

## Instalace

```bash
# kód v ~/voice-trainer, data a venvy na /mnt/models/voice-trainer
sudo apt install -y espeak-ng ffmpeg        # espeak-ng nutný pro fonémizaci
python3 -m venv /mnt/models/voice-trainer/venv
/mnt/models/voice-trainer/venv/bin/pip install numpy PyYAML
# piper tréninkové venv (velké, ~6 GB):
python3 -m venv /mnt/models/voice-trainer/venv-piper
/mnt/models/voice-trainer/venv-piper/bin/pip install torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118
/mnt/models/voice-trainer/venv-piper/bin/pip install \
  -e '/mnt/models/voice-trainer/piper1-gpl/repo[train]'
cd /mnt/models/voice-trainer/piper1-gpl/repo && ./build_monotonic_align.sh
cp ~/voice-trainer/config.example.yaml ~/voice-trainer/config.yaml  # upravit
~/voice-trainer/bin/voice-trainer init
```

Jiné umístění venv nastavíš proměnnou `VOICE_TRAINER_VENV`.

## Jak vzniká dataset

- LLM vygeneruje dávku vět (default 10), klient je dostává po jedné.
- STT porovná nahrávku s předlohou; **≥ 90 %** = PASS. Jinak se nahrává stejná věta znovu.
- Smazaná nahrávka jde do `dataset/trash/<mluvčí>/` a její věta se vrací na začátek fronty.
- Po každých N PASS (default 10) se **automaticky spustí trénink** (detached).
- Import hromadných vět: `voice-trainer import-text --file vety.txt`
  (formát `000001|věta` nebo čisté řádky).

## Jazyky

Jazyk patří k hlasu a určuje generování vět, kontrolu přes STT i fonémizaci při tréninku.
Vestavěné jsou `cs` a `en`, další se doplní do `languages:` v `config.yaml`
nebo do `voice_trainer/languages.py`.

```bash
voice-trainer new-speaker --name anna --language en
voice-trainer speaker-language --name anna --language cs   # jen dokud hlas nemá nahrávky
voice-trainer languages                                    # JSON, has_base = lze trénovat
voice-trainer generate --language en
voice-trainer import-text --file sentences.txt --language en
```

Trénink potřebuje základní model ve stejném jazyce (`training.bases.<název>.language`,
výchozí `cs`). Bez něj jde nahrávat, ale trénink skončí chybou.

## HTTP API

API poslouchá jen na `api_host:api_port`. Nastav adresu v soukromé síti (např. ZeroTier),
provoz API sám nešifruje.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/voice-trainer-api.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voice-trainer-api
sudo loginctl enable-linger "$USER"             # běh i bez přihlášení a po restartu
cat /mnt/models/voice-trainer/api-token         # token pro klienta
```

## Kontrola a review

```bash
voice-trainer dataset        # info (JSON)
voice-trainer check --id 1   # znovu STT + quality jedné nahrávky
voice-trainer review         # list REVIEW/FAIL/SKIP
voice-trainer review --set 12 PASS   # ruční překlasifikace
voice-trainer quality        # quality report (clipping, RMS, duplikáty, STT)
```

## Trénink

```bash
voice-trainer train                 # 1. trénink (base = jirka-medium ckpt)
voice-trainer train --yes           # bez potvrzení (používá auto-trigger)
voice-trainer resume                # pokračovat z posledního checkpointu
voice-trainer train --checkpoint training/run_001/.../last.ckpt
voice-trainer status                # průběh (JSON) — epoch, loss, GPU
voice-trainer runs                  # runy + checkpointy (append-only)
```

- **Kumulace:** každý run trénuje na VŠECHNY PASS nahrávky k danému datu
  (dataset v2 = v1 + nové). Checkpoint z runu N je base pro run N+1.
- Trénink běží detached (setsid) — zavření ssh terminálu ho nezastaví.
- Služby v `gpu_lane_units` se před tréninkem zastaví přes `sudo -n`. Po tréninku se vrátí jen ty,
  které jsou i v `gpu_restart_units`. Když pak na GPU zbývá méně než `min_free_gpu_mb`, run hned skončí chybou.
- Z klienta: `POST /training/start` a `/training/stop`, průběh přes `GET /training`. Zastavený run
  si nechá `last.ckpt` a další na něj naváže. Aby trénink přežil restart API, má jednotka `KillMode=process`.
- Checkpointy: Lightning top-5 dle `val_mel` + `last.ckpt` per run dir.
  Starší checkpointy se nikdy nemažou.

## Export + test

```bash
voice-trainer export                    # poslední checkpoint → export/*.onnx
voice-trainer export --list
voice-trainer test                      # test.wav z posledního exportu
voice-trainer test --model export/xxx.onnx --text "Moje testovací věta."
```

Výstup: `export/<jméno>-medium.onnx` + `.onnx.json` — hotový Piper model.

Z klienta: `POST /training/test` vrátí větu namluvenou posledním uloženým stavem hlasu, i během tréninku.
Model pro tenhle účel je v `export/app/` a syntéza běží na serveru bez GPU.

Kam se hotový model nasazuje, je věc konkrétní instalace. `POST /training/publish` proto jen
spustí `publish_command` z `config.yaml`, například:

```bash
#!/bin/sh
set -e
rsync "$VT_MODEL" "tts-host:voices/$VT_SPEAKER.onnx"
rsync "$VT_MODEL_CONFIG" "tts-host:voices/$VT_SPEAKER.onnx.json"
```

## Robustnost

- Nahrávka se ukládá nejdřív do `incoming/`, po STT kontrole přesun do
  `dataset/wav/` (atomic rename) + DB insert.
- Přerušené nahrávání: rozpracovaná session zůstává aktivní a další spuštění na ni naváže.
- Trénink: `status.json` + `training.log` per run; kill procesu nezničí DB.
- Výpadek GPU: runner označí run `failed`, metadata zůstanou.
- Logy: `<data_root>/logs/{recording,stt,training,system}.log`.

## Troubleshooting

**GPU:** `voice-trainer status` ukazuje free VRAM. Na GPU řady Pascal musí být `torch`
build cu118 (`torch.cuda.get_device_name(0)`) a DDP s `NCCL_P2P_DISABLE=1`.

**STT:** první požadavek po startu serveru může trvat až 120 s, proto `stt_timeout` a `stt_retries`.

**LLM:** ověř `curl <llm_url>/api/tags`.

**Trénink selže hned:** chybí `espeak-ng` (apt), nebo Lightning checkpoint
gotchy (medium quality jen; případně clean_checkpoint fix z
OHF-Voice/piper1-gpl#132) — viz `training/run_XXX/training.log`.

## Licence

Kód v tomto repozitáři je pod licencí [MIT](LICENSE). Piper (piper1-gpl) tu není přibalený,
instaluje se zvlášť a platí pro něj jeho vlastní licence GPL. `voice_trainer/piper_fit.py`
a `voice_trainer/clean_ckpt.py` ho za běhu importují, takže kdo šíří celek i s Piperem,
musí pro něj dodržet podmínky GPL.
