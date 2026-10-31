"""voice-trainer CLI — dataset, trénink, export, JSON protokol pro klienta.

Profily mluvčích (2026-09-26): nahrávání se váže JEN na mluvčího (profil);
base model se volí až při tréninku (dvojice mluvčí × base = vlastní chain).
"""
import argparse
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_trainer import dataset as ds     # noqa: E402
from voice_trainer import db as dbmod       # noqa: E402
from voice_trainer import llm as llm_mod    # noqa: E402
from voice_trainer import quality as qmod   # noqa: E402
from voice_trainer import stt as stt_mod    # noqa: E402
from voice_trainer import training as tr    # noqa: E402
from voice_trainer.config import load       # noqa: E402

log = logging.getLogger("voice-trainer")


def setup_logging(cfg, name="system"):
    logdir = Path(cfg["data_root"]) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logdir / f"{name}.log"),
                  logging.StreamHandler(sys.stderr)])


def get_db(cfg):
    return dbmod.Db(Path(cfg["data_root"]) / "voice-trainer.db")


def eprint(*a):
    print(*a, file=sys.stderr)


def spawn_llm_gen(cfg):
    """Detached pozadí generování vět (voice_trainer.llm_gen); flock uvnitř
    modulu chrání před duplicitním spuštěním."""
    log_path = Path(cfg["data_root"]) / "logs" / "llm-gen.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as lf:
        subprocess.Popen(
            [sys.executable, "-m", "voice_trainer.llm_gen"],
            stdout=lf, stderr=subprocess.STDOUT,
            cwd=cfg["repo_dir"], start_new_session=True)


def resolve_speaker(db, name=None):
    """Jméno/id mluvčího → speaker_id (rowid). name None → None (auto)."""
    if name is None:
        return None
    sid = db.speaker_id(str(name))
    if sid is None:
        raise SystemExit(f"neznámý mluvčí: {name} (voice-trainer speakers)")
    return sid


def pick_speaker(db, args_speaker):
    """Interaktivní výběr mluvčího, když jich je víc a není zadaný.
    Prázdné profily (bez nahrávek) se ignorují — neblokují trénink."""
    if args_speaker:
        return resolve_speaker(db, args_speaker)
    sps = [s for s in db.speakers() if db.speaker_stats(s["id"])[0] > 0]
    if not sps:
        raise SystemExit("dataset nemá žádné nahrávky — nejdřív nahrávej")
    if len(sps) == 1:
        return sps[0]["id"]
    eprint("mluvčí:")
    for s in sps:
        p, d = db.speaker_stats(s["id"])
        eprint(f"  [{s['id']}] {s['name']} — {p} PASS, {_hms(d)}")
    ans = input("mluvčí [název/id]: ").strip()
    if not ans:
        raise SystemExit("zrušeno")
    if ans.isdigit():
        sp = db.speaker(int(ans))
        if sp is None:
            raise SystemExit(f"neznámé id mluvčího: {ans}")
        return sp["id"]
    return resolve_speaker(db, ans)


def pick_base(cfg, args_base):
    """Interaktivní výběr base modelu (label z training.bases)."""
    bs = tr.bases(cfg)
    if not bs:
        return None
    if args_base:
        if args_base not in bs:
            raise SystemExit(f"neznámý base model: {args_base} "
                             f"(dostupné: {', '.join(bs)})")
        return args_base
    if len(bs) == 1:
        return next(iter(bs))
    eprint("base modely:")
    for label, b in bs.items():
        eprint(f"  {label} — {b['ckpt']}")
    ans = input("base [název]: ").strip()
    if not ans:
        raise SystemExit("zrušeno")
    if ans not in bs:
        raise SystemExit(f"neznámý base model: {ans}")
    return ans


# ── dataset ──────────────────────────────────────────────────────────────────
def cmd_init(cfg, args):
    setup_logging(cfg)
    db = get_db(cfg)
    for p in (ds.wav_dir(cfg), ds.master_dir(cfg), ds.incoming_dir(cfg),
              ds.training_root(cfg)):
        Path(p).mkdir(parents=True, exist_ok=True)
    n = ds.write_metadata(db, cfg)
    gpus = tr.gpu_status()
    eprint(f"✓ init: {Path(cfg['data_root'])}")
    eprint(f"  DB: {Path(cfg['data_root']) / 'voice-trainer.db'}")
    eprint(f"  metadata (všechny mluvčí): {n} PASS nahrávek")
    eprint(f"  GPU: " + ", ".join(
        f"[{g['index']}] {g['name']} {g['free_mb']}/{g['total_mb']} MiB" for g in gpus))


def cmd_speakers(cfg, args):
    """Seznam mluvčích + statistiky (JSON na stdout — čte ho klient)."""
    db = get_db(cfg)
    bs = tr.bases(cfg)
    out = []
    for s in db.speakers():
        p, d = db.speaker_stats(s["id"])
        pending = len(db.unrecorded_utterances(s["id"]))
        out.append({"id": s["id"], "name": s["name"], "passes": p,
                    "duration_s": round(d, 1), "duration_hms": _hms(d),
                    "pending": pending,
                    "created_at": s["created_at"]})
    print(json.dumps({"speakers": out,
                      "base_models": list(bs.keys())},
                     ensure_ascii=False, indent=2))


def cmd_new_speaker(cfg, args):
    """Vytvoří profil mluvčího (adresáře + DB řádek)."""
    db = get_db(cfg)
    sid = db.ensure_speaker(args.name)
    sp = db.speaker(sid)
    ds.wav_dir(cfg, sp["name"]).mkdir(parents=True, exist_ok=True)
    ds.master_dir(cfg, sp["name"]).mkdir(parents=True, exist_ok=True)
    eprint(f"✓ mluvčí '{sp['name']}' (id {sid}) připraven")
    print(json.dumps({"id": sid, "name": sp["name"]}, ensure_ascii=False))


def cmd_import_text(cfg, args):
    setup_logging(cfg, "recording")
    db = get_db(cfg)
    path = Path(args.file)
    text = path.read_text(encoding="utf-8-sig")
    added, skipped = 0, 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # formát 000001|text → ignorovat ID (má vlastní DB id)
        if "|" in line:
            parts = line.split("|", 1)
            line = parts[1].strip() if len(parts) > 1 else parts[0].strip()
        uid = db.add_utterance(line, "import")
        if uid and db.utterance(uid)["created_at"]:
            skipped += 1 if False else 0  # dedup je v add_utterance
        skipped += 0
    # přesné počítání: kolik řádků šlo do DB
    total = db.one("SELECT COUNT(*) n FROM utterances")["n"]
    eprint(f"✓ import: {path} → {total} celkem vět v datasetu")
    log.info("import-text %s (celkem %d vět)", path, total)
    ds.write_metadata(db, cfg)


def cmd_generate(cfg, args):
    """Vygeneruje vět přes LLM (ollama glm) a uloží do datasetu."""
    setup_logging(cfg, "system")
    db = get_db(cfg)
    count = args.count or int(cfg["llm_batch_size"])
    eprint(f"» generuji {count} vět ({cfg['llm_model']} @ {cfg['llm_url']})…")
    sentences = llm_mod.generate(cfg, count)
    added = 0
    for s in sentences:
        db.add_utterance(s, "llm")
        added += 1
    total = db.one("SELECT COUNT(*) n FROM utterances")["n"]
    eprint(f"✓ vygenerováno {added} vět (dataset má {total})")
    for s in sentences[:5]:
        eprint(f"  • {s}")
    if len(sentences) < added and added < count:
        eprint(f"⚠ LLM vrátil jen {added}/{count} platných vět")


def cmd_dataset(cfg, args):
    db = get_db(cfg)
    passes, dur = db.stats()
    utts = db.one("SELECT COUNT(*) n FROM utterances")["n"]
    sess = db.all("SELECT * FROM sessions ORDER BY id DESC LIMIT 10")
    runs = db.all("SELECT name, status, started_at, speaker_id, base_name "
                  "FROM runs ORDER BY id DESC LIMIT 5")
    sr = ds.base_sample_rate(cfg)
    sps = []
    for s in db.speakers():
        p, d = db.speaker_stats(s["id"])
        sps.append({"id": s["id"], "name": s["name"], "passes": p,
                    "duration_s": round(d, 1), "duration_hms": _hms(d)})
    out = {
        "utterances": utts, "pass_recordings": passes,
        "duration_s": round(dur, 1), "duration_hms": _hms(dur),
        "sample_rate": sr, "speakers": sps,
        "sessions": [dict(s) for s in sess],
        "runs": [dict(r) for r in runs],
        "pending": len(db.unrecorded_utterances()),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _hms(sec):
    sec = int(sec or 0)
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m {sec % 60}s"


# ── protokol pro klienta (JSON na stdout) ────────────────────────────────────
def cmd_next_sentence(cfg, args):
    """Vrátí další větu k nahrání PRO DANÉHO MLUVČÍHO. Vytváří/pokračuje
    session (vázaná na mluvího); když je dataset 'prázdný', vygeneruje dávku
    přes LLM. Po N PASS spustí trénink (default base)."""
    setup_logging(cfg, "recording")
    db = get_db(cfg)
    batch = int(cfg["llm_batch_size"])
    speaker_name = args.speaker
    speaker_id = resolve_speaker(db, speaker_name)
    sid = args.session
    if sid:
        s = db.session(sid)
        if s is None:
            raise SystemExit(f"neznámá session: {sid}")
        # speaker bere ze session (klient si ho zvolil při startu)
        if s["speaker_id"]:
            sp = db.speaker(s["speaker_id"])
            speaker_name = sp["name"] if sp else speaker_name
            speaker_id = s["speaker_id"]
    else:
        if speaker_id is None:
            sps = db.speakers()
            if not sps:
                raise SystemExit("žádný mluvčí — nejdřív voice-trainer new-speaker")
            if len(sps) > 1:
                raise SystemExit("více mluvčích — klient musí poslat --speaker")
            speaker_id = sps[0]["id"]
            speaker_name = sps[0]["name"]
        active = db.active_sessions(speaker_id)
        if active:
            sid = active[0]["id"]
        else:
            sid = db.create_session(batch, speaker_id=speaker_id)
            eprint(f"(nová session: {sid}, mluvčí: {speaker_name})")

    # trénink trigger: po každých N PASS v této session
    sess = db.session(sid)
    pass_count = db.one(
        "SELECT COUNT(*) n FROM recordings WHERE session_id=? AND status='PASS'",
        (sid,))["n"]
    training_info = None
    # auto_train / train_every: trigger po N PASS (nezavislý na velikosti
    # generování vět llm_batch_size); train_every 0 nebo auto_train false
    # → žádné nové runy (jen ruční start / dokončení běžícího runu)
    train_every = int(cfg.get("train_every") or 0)
    if cfg.get("auto_train", True) and train_every > 0 and pass_count and \
            pass_count % train_every == 0 and \
            sess["status"] == "active":
        # spustit trénink a session uzavřít (další = nová session)
        # když už fit běží → run se zařadí do fronty (spustí se po dokončení)
        defer = tr.training_running(cfg)
        eprint(f"» {pass_count} PASS → "
               f"{'zařazuji trénink do fronty (GPU obsazená)…' if defer else 'spouštím trénink…'}")
        try:
            training_info = tr.start_training(db, cfg, defer=defer,
                                              speaker_id=speaker_id)
            db.finish_session(sid)
            log.info("training spuštěn po %d PASS (%s, mluvčí %s)",
                     pass_count, sid, speaker_name)
        except Exception as e:
            eprint(f"✗ trénink se nepodařilo spustit: {e}")
            training_info = {"error": str(e)}
        # hned otevřít novou session — jinak by první PASS po triggeru
        # spadl do uzavřené session a čítač dávky by se posunul
        sid = db.create_session(batch, speaker_id=speaker_id)
        eprint(f"(nová session: {sid})")
        sess = db.session(sid)
        pass_count = 0

    # další věta: první bez PASS pro tohoto mluvčího; když zbyvá málo →
    # předgenerovat na pozadí (LLM občas generuje i minuty — hranice dávky
    # nesmí zastavit nahrávání)
    pending = db.unrecorded_utterances(speaker_id)
    if len(pending) <= 3:
        spawn_llm_gen(cfg)
    if not pending:
        eprint("» čekám na nové věty (generuji na pozadí)…")
        deadline = datetime.datetime.now() + datetime.timedelta(seconds=600)
        while datetime.datetime.now() < deadline:
            time.sleep(2)
            pending = db.unrecorded_utterances(speaker_id)
            if pending:
                break
        if not pending:
            raise SystemExit("LLM nevygeneroval žádné nové věty (timeout 600 s)")
    utt = pending[0]
    total_pass, total_dur = db.stats(speaker_id)
    print(json.dumps({
        "session": sid,
        "session_status": sess["status"],
        "speaker": speaker_name,
        "utterance_id": utt["id"],
        "text": utt["text"],
        "session_pass_count": pass_count,
        "batch_size": int(batch),
        "train_every": train_every,
        "auto_train": bool(cfg.get("auto_train", True) and train_every > 0),
        "dataset_passes": total_pass,
        "dataset_duration_s": round(total_dur, 1),
        "training": training_info,
    }, ensure_ascii=False))


def cmd_skip(cfg, args):
    setup_logging(cfg, "recording")
    db = get_db(cfg)
    s = db.session(args.session)
    speaker_id = s["speaker_id"] if s else None
    db.add_recording(int(args.utterance), args.session, None, "SKIP", None, None,
                     None, None, None, note=args.note or "skip během nahrávání",
                     speaker_id=speaker_id)
    eprint(f"✓ SKIP věta {args.utterance}")


def cmd_save(cfg, args):
    """Uloží nahrané WAV (přes incoming) → dataset + DB. Atomic: tmp→rename.
    Klient ověřil STT; server přepočítá quality a případně označí REVIEW.
    Speaker bere ze session (klient si ho zvolil při startu)."""
    setup_logging(cfg, "recording")
    db = get_db(cfg)
    src = Path(args.file)
    if not src.exists():
        raise SystemExit(f"soubor nenalezen: {src}")
    s = db.session(args.session)
    if s is None:
        raise SystemExit(f"neznámá session: {args.session}")
    speaker_id = s["speaker_id"]
    sp = db.speaker(speaker_id) if speaker_id else None
    sp_name = sp["name"] if sp else None
    sr_target = ds.base_sample_rate(cfg)
    name = ds.utterance_name(int(args.utterance))
    wav_dst = ds.wav_dir(cfg, sp_name) / name
    wav_dst.parent.mkdir(parents=True, exist_ok=True)
    master_dst = ds.master_dir(cfg, sp_name) / name
    master_dst.parent.mkdir(parents=True, exist_ok=True)

    # master (raw) nejprve do tmp, po konverzi atomic replace do datasetu;
    # klient uploaduje FLAC (lossless, rychlejší přes ZeroTier) → převést
    if src.suffix.lower() == ".flac":
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(src), "-c:a", "pcm_s16le",
             str(master_dst.with_suffix(".tmp.wav"))],
            check=True, timeout=120)
    else:
        shutil.copy2(src, master_dst.with_suffix(".tmp.wav"))
    conv = master_dst.with_suffix(".tmp2.wav")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-ar", str(sr_target), "-ac", "1", "-c:a", "pcm_s16le", str(conv)],
        check=True, timeout=120)
    if wav_dst.exists():  # RETAKE přepisuje (předešlé zůstane v master/)
        shutil.copy2(wav_dst, ds.master_dir(cfg, sp_name) / f"{name}.old")
    conv.replace(wav_dst)
    master_dst.with_suffix(".tmp.wav").replace(master_dst)
    info = ds.check_wav(wav_dst)
    sr, ch, dur = info if info else (sr_target, 1, 0.0)
    master_rel = (f"master/{sp_name}/{name}" if sp_name
                  else f"master/{name}")

    rec = db.add_recording(
        int(args.utterance), args.session, name, "PASS",
        args.stt_text, float(args.similarity) if args.similarity else None,
        dur, sr, ch, master_wav=master_rel,
        recorded_at=args.recorded_at, speaker_id=speaker_id)
    # quality → REVIEW pokud podezřelé (nemazat!)
    issues, ok = qmod.check_recording(db, cfg, dict(rec))
    if not ok:
        db.update_recording_status(rec["id"], "REVIEW", note="; ".join(issues))
        rec = db.recording(rec["id"])
    pass_count = db.one(
        "SELECT COUNT(*) n FROM recordings WHERE session_id=? AND status='PASS'",
        (args.session,))["n"]
    log.info("saved %s session=%s mluvčí=%s sim=%s issues=%s", name,
             args.session, sp_name, args.similarity, issues)
    print(json.dumps({
        "recording_id": rec["id"], "status": rec["status"],
        "speaker": sp_name,
        "quality_issues": issues, "duration": round(dur, 2),
        "sample_rate": sr, "session_pass_count": pass_count,
    }, ensure_ascii=False))


def cmd_check(cfg, args):
    """Ruční/automatická kontrola: znovu STT + quality na uloženou nahrávku."""
    setup_logging(cfg, "stt")
    db = get_db(cfg)
    rec = db.recording(int(args.id))
    if rec is None:
        raise SystemExit(f"neznámá nahrávka: {args.id}")
    path = ds.recording_path(db, cfg, rec)
    if not path.exists():
        raise SystemExit(f"wav nenalezeno: {path}")
    eprint(f"» STT {path} …")
    text = stt_mod.transcribe(path, cfg)
    utt = db.utterance(rec["utterance_id"])
    sim = stt_mod.similarity(utt["text"], text) if utt else None
    db.q("UPDATE recordings SET stt_text=?, similarity=?, checked_at=? WHERE id=?",
         (text, sim, dbmod.now(), rec["id"]))
    issues, ok = qmod.check_recording(db, cfg, dict(rec))
    eprint(f"  reference: {utt['text']}")
    eprint(f"  STT:       {text}")
    eprint(f"  podobnost: {sim:.3f}" if sim is not None else "  podobnost: —")
    eprint(f"  quality:   {issues or 'OK'}")
    print(json.dumps({"recording": rec["id"], "stt": text, "similarity": sim,
                      "issues": issues, "ok": ok}, ensure_ascii=False))


def cmd_review(cfg, args):
    setup_logging(cfg, "recording")
    db = get_db(cfg)
    if args.set is None:
        rows = db.all(
            "SELECT r.*, s.name sp_name FROM recordings r "
            "LEFT JOIN speakers s ON s.id = r.speaker_id "
            "WHERE r.status IN ('REVIEW','FAIL','RETAKE','SKIP') ORDER BY r.id")
        for r in rows:
            utt = db.utterance(r["utterance_id"])
            eprint(f"[{r['id']}] {r['status']:6} {r['sp_name'] or '?':10} "
                   f"věta {r['utterance_id']} sim={r['similarity']} "
                   f"| {utt['text'] if utt else '?'}")
            if r["note"]:
                eprint(f"        note: {r['note']}")
        return
    rid, status = args.set
    if status not in ("PASS", "FAIL", "RETAKE", "SKIP", "REVIEW"):
        raise SystemExit(f"neplatný status: {status}")
    db.update_recording_status(int(rid), status)
    eprint(f"✓ nahrávka {rid} → {status}")
    ds.write_metadata(db, cfg)


def cmd_quality(cfg, args):
    db = get_db(cfg)
    speaker_id = resolve_speaker(db, args.speaker) if args.speaker else None
    rows = qmod.run_report(db, cfg, speaker_id)
    bad = [r for r in rows if r["issues"]]
    eprint(f"pass nahrávek: {len(rows)}, podezřelé: {len(bad)}")
    for r in bad:
        utt = db.utterance(r["utterance"])
        eprint(f"[{r['id']}] {(r.get('speaker') or '?'):8} věta {r['utterance']}: "
               f"{', '.join(r['issues'])}")
        eprint(f"        {utt['text'] if utt else '?'}")
    print(json.dumps(rows, ensure_ascii=False))


# ── trénink ──────────────────────────────────────────────────────────────────
def _confirm_gpu(cfg, yes):
    gpus = tr.gpu_status()
    for g in gpus:
        eprint(f"GPU {g['index']}: {g['name']} — free {g['free_mb']}/{g['total_mb']} MiB")
    states = tr.lane_states(cfg)
    active = [u for u, s in states.items() if s == "active"]
    if active:
        eprint(f"GPU lanes běží: {', '.join(active)} — budou graceful stopnuty")
    if not yes:
        ans = input("Pokračovat a stopnout lanes? [a/N] ").strip().lower()
        if ans not in ("a", "y", "ano", "yes"):
            raise SystemExit("zrušeno")
    return gpus


def cmd_train(cfg, args):
    setup_logging(cfg, "training")
    db = get_db(cfg)
    if tr.training_running(cfg):
        raise SystemExit("trénink už běží (pgrep piper.train) — voice-trainer status")
    _confirm_gpu(cfg, args.yes)
    speaker_id = pick_speaker(db, args.speaker)
    base_name = pick_base(cfg, args.base)
    info = tr.start_training(db, cfg, base_ckpt=args.checkpoint,
                             max_epochs=args.epochs, run_name=args.run,
                             speaker_id=speaker_id, base_name=base_name)
    eprint(f"✓ trénink {info['run']} spuštěn (detached — terminál můžeš zavřít)")
    eprint(f"  mluvčí:  {info['speaker']}")
    eprint(f"  base:    {info['base_name']} ({info['base']})")
    eprint(f"  dataset: {info['samples']} samples, {info['duration_s']}s, "
           f"sr={info['sample_rate']}")
    eprint(f"  status:  voice-trainer status")
    log.info("training launched: %s", info)


def cmd_status(cfg, args):
    db = get_db(cfg)
    speaker_id = resolve_speaker(db, args.speaker) if args.speaker else None
    run = db.latest_run(speaker_id=speaker_id,
                        base_name=args.base if args.base else None)
    out = {"training_running": tr.training_running(cfg)}
    if run:
        sp = db.speaker(run["speaker_id"]) if run["speaker_id"] else None
        out["speaker"] = sp["name"] if sp else None
        out["base_name"] = run["base_name"]
        run_dir = Path(run["dataset_dir"])
        status = {}
        try:
            status = json.load(open(run_dir / "status.json"))
        except (OSError, ValueError):
            pass
        out["run"] = run["name"]
        out["run_status"] = run["status"]
        out.update(status)
        # lightning metrics (poslední epoch/loss/val_mel)
        metrics = _latest_metrics(run_dir)
        out["metrics"] = metrics
        # tail logu
        try:
            lines = (run_dir / "training.log").read_text().splitlines()
            out["log_tail"] = lines[-5:]
        except OSError:
            pass
    passes, dur = db.stats(speaker_id)
    out["dataset"] = {"pass_recordings": passes, "duration_s": round(dur, 1),
                      "duration_hms": _hms(dur)}
    out["gpus"] = tr.gpu_status()
    out["lanes"] = tr.lane_states(cfg)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _latest_metrics(run_dir):
    import csv
    best = None
    for f in Path(run_dir).rglob("metrics.csv"):
        try:
            rows = list(csv.reader(open(f, errors="replace")))
        except OSError:
            continue
        if not rows:
            continue
        hdr = rows[0]
        best = rows[-1]  # poslední řádek
    if not best:
        return None
    out = {}
    try:
        hdr = [h.strip() for h in open(
            str(list(Path(run_dir).rglob("metrics.csv"))[0]), errors="replace"
        ).readline().strip().split(",")]
        for k, v in zip(hdr, best):
            if v:
                out[k] = v
    except Exception:
        pass
    return out


def _run_for_ckpt(db, ckpt):
    """Run patřící k checkpointu (dataset_dir je rodič cesty)."""
    p = Path(ckpt).absolute()
    for r in db.all("SELECT * FROM runs"):
        d = Path(r["dataset_dir"] or "/nonexistent")
        if d in p.parents or d == p.parent:
            return r
    return None


def cmd_export(cfg, args):
    setup_logging(cfg, "training")
    db = get_db(cfg)
    if args.list:
        exp = Path(cfg["data_root"]) / "export"
        if exp.exists():
            for p in sorted(exp.glob("*.onnx")):
                eprint(f"  {p}")
        else:
            eprint("(žádné exporty)")
        return
    speaker_id = resolve_speaker(db, args.speaker) if args.speaker else None
    ckpt = args.checkpoint
    if not ckpt:
        ckpt = tr.find_checkpoint(db, cfg, "last", speaker_id,
                                  args.base if args.base else None)
    if not ckpt or not Path(ckpt).exists():
        raise SystemExit("checkpoint nenalezen — zadej --checkpoint PATH "
                         "nebo --speaker/--base")
    # default název: <mluvčí>-<base>-<timestamp> (z runu checkpointu)
    run = _run_for_ckpt(db, ckpt)
    if args.name:
        name = args.name
    else:
        sp = db.speaker(run["speaker_id"]) if run and run["speaker_id"] else None
        base_label = run["base_name"] if run else (args.base or "model")
        sp_label = sp["name"] if sp else (args.speaker or "hlas")
        name = datetime.datetime.now().strftime(
            f"{sp_label}-{base_label}-%Y%m%d-%H%M")
    outdir = Path(cfg["data_root"]) / "export"
    outdir.mkdir(parents=True, exist_ok=True)
    onnx = outdir / f"{name}.onnx"
    eprint(f"» export {ckpt} → {onnx}")
    tr.export_onnx(cfg, ckpt, str(onnx))
    # config.json z run dir → .onnx.json (checkpoint bývá hluboko v
    # lightning_logs → hledej config.json směrem nahoru)
    src = next((p for p in Path(ckpt).absolute().parents
                if (p / "config.json").exists()), None)
    if src:
        shutil.copy2(src / "config.json", str(onnx) + ".json")
    else:
        eprint("⚠ config.json k checkpointu nenalezen — .onnx.json chybí!")
    eprint(f"✓ export: {onnx} (+ .json)")


def cmd_test(cfg, args):
    setup_logging(cfg, "training")
    db = get_db(cfg)
    model = args.model
    if not model:
        exp = sorted((Path(cfg["data_root"]) / "export").glob("*.onnx"),
                     key=lambda p: p.stat().st_mtime, reverse=True) \
            if (Path(cfg["data_root"]) / "export").exists() else []
        if exp:
            model = str(exp[0])
    if not model:
        raise SystemExit("žádný model — nejdřív voice-trainer export")
    outdir = Path(model).parent
    text = args.text or "Ahoj, tohle je test mého vlastního Piper hlasu."
    stamp = datetime.datetime.now().strftime("%H%M%S")
    out = outdir / f"test_{Path(model).stem}_{stamp}.wav"
    eprint(f"» inference {model}")
    tr.test_inference(cfg, model, text, out)
    eprint(f"✓ {out}")
    log.info("test inference %s → %s", model, out)


def cmd_runs(cfg, args):
    db = get_db(cfg)
    for r in db.all(
            "SELECT r.*, s.name sp_name FROM runs r "
            "LEFT JOIN speakers s ON s.id = r.speaker_id ORDER BY r.id"):
        eprint(f"[{r['name']}] {r['status']:8} mluvčí={r['sp_name'] or '?'} "
               f"base={r['base_name'] or '?'} ({r['base_ckpt']}) "
               f"started={r['started_at']} dir={r['dataset_dir']}")
        cdir = Path(r["dataset_dir"])
        if cdir.exists():
            for c in sorted(cdir.rglob("*.ckpt")):
                eprint(f"    {c.relative_to(cdir)}")


def cmd_api(cfg, args):
    """HTTP API pro iOS aplikaci (běží trvale)."""
    from voice_trainer import api
    setup_logging(cfg, "api")
    api.serve(cfg)


def main():
    parser = argparse.ArgumentParser(
        prog="voice-trainer",
        description="Trénink vlastního českého Piper hlasu")
    parser.add_argument("--config", help="cesta ke config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="vytvořit adresáře/DB")
    sub.add_parser("speakers", help="seznam mluvčích (JSON)")
    p = sub.add_parser("new-speaker", help="vytvořit profil mluvčího")
    p.add_argument("--name", required=True)
    p = sub.add_parser("import-text", help="import TXT/CSV vět (id|text nebo čisté řádky)")
    p.add_argument("--file", required=True)
    p = sub.add_parser("generate", help="vygenerovat vět LLM (ollama)")
    p.add_argument("--count", type=int, default=None)
    sub.add_parser("dataset", help="info o datasetu (JSON)")
    p = sub.add_parser("next-sentence", help="[protokol] další věta k nahrání (JSON)")
    p.add_argument("--session", default=None, help="pokračovat v konkrétní session")
    p.add_argument("--speaker", default=None,
                   help="mluvčí (profil) — při nové session")
    p = sub.add_parser("skip")
    p.add_argument("--session", required=True)
    p.add_argument("--utterance", required=True)
    p.add_argument("--note", default=None)
    p = sub.add_parser("save")
    p.add_argument("--session", required=True)
    p.add_argument("--utterance", required=True)
    p.add_argument("--file", required=True)
    p.add_argument("--stt-text", default="")
    p.add_argument("--similarity", default=None)
    p.add_argument("--recorded-at", default=None)
    p = sub.add_parser("check")
    p.add_argument("--id", required=True)
    p = sub.add_parser("review")
    p.add_argument("--set", nargs=2, metavar=("ID", "STATUS"), default=None)
    p = sub.add_parser("quality", help="quality report (JSON)")
    p.add_argument("--speaker", default=None)
    p = sub.add_parser("train")
    p.add_argument("--checkpoint", default=None, help="base checkpoint (resume)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--speaker", default=None, help="mluvčí (profil) dat")
    p.add_argument("--base", default=None,
                   help="base model (label z training.bases)")
    p.add_argument("--yes", action="store_true", help="bez potvrzení (auto po N větách)")
    # resume = train bez explicitního checkpointu (chain per mluvčí × base)
    p = sub.add_parser("resume", help="pokračovat z posledního checkpointu")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--speaker", default=None)
    p.add_argument("--base", default=None)
    p.add_argument("--yes", action="store_true")
    p = sub.add_parser("status", help="status tréninku (JSON)")
    p.add_argument("--speaker", default=None)
    p.add_argument("--base", default=None)
    sub.add_parser("runs", help="seznam tréninků + checkpointů")
    p = sub.add_parser("export")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--speaker", default=None)
    p.add_argument("--base", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--list", action="store_true")
    p = sub.add_parser("test")
    p.add_argument("--model", default=None)
    p.add_argument("--text", default=None)
    sub.add_parser("api", help="HTTP API pro iOS aplikaci")

    args = parser.parse_args()
    cfg = load(args.config)
    if args.cmd == "resume":
        args.cmd = "train"

    handlers = {
        "init": cmd_init, "speakers": cmd_speakers,
        "new-speaker": cmd_new_speaker,
        "import-text": cmd_import_text,
        "generate": cmd_generate, "dataset": cmd_dataset,
        "next-sentence": cmd_next_sentence, "save": cmd_save,
        "skip": cmd_skip, "check": cmd_check, "review": cmd_review,
        "quality": cmd_quality, "train": cmd_train,
        "status": cmd_status, "export": cmd_export, "test": cmd_test,
        "runs": cmd_runs, "api": cmd_api,
    }
    handlers[args.cmd](cfg, args)


if __name__ == "__main__":
    main()