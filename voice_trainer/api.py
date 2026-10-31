"""HTTP API pro klientské aplikace (přes ZeroTier) — bez závislostí navíc.

Protokol nahrávání jde přes stejné příkazy jako terminálový klient (voice-trainer
next-sentence/save/skip), takže logika session a tréninku je na jednom místě.
Přístup hlídá Bearer token ze souboru <data_root>/api-token; ZeroTier šifruje provoz.

  GET    /health                     stav serveru (i kontrola tokenu)
  GET    /speakers                   mluvčí + statistiky
  POST   /speakers                   {"name": …, "language": …} nový mluvčí
  POST   /speakers/<jméno>/language  {"language": …} změna jazyka hlasu bez nahrávek
  GET    /languages                  dostupné jazyky
  GET    /next?speaker=…             další věta k nahrání
  POST   /transcribe?utterance=…     tělo = audio; STT + shoda, nic neukládá do datasetu
  POST   /save                       {"session","utterance","upload_id"} uloží PASS
  POST   /discard                    {"upload_id"} zahodí nahrávku (znovu)
  POST   /skip                       {"session","utterance"} přeskočí větu
  GET    /recordings?speaker=…&q=…&offset=…&limit=…
  GET    /recordings/<id>/audio      audio/flac
  DELETE /recordings/<id>            přesune soubory do dataset/trash, věta jde zpět do fronty
"""
import datetime
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from voice_trainer import dataset as ds
from voice_trainer import db as dbmod
from voice_trainer import languages
from voice_trainer import stt as stt_mod
from voice_trainer import training as tr

log = logging.getLogger("voice-trainer.api")

MAX_UPLOAD = 40 * 1024 * 1024
UPLOAD_ID = re.compile(r"^app_[0-9a-f]{24}$")
SPEAKER_NAME = re.compile(r"^[\w -]{1,40}$", re.UNICODE)
UPLOAD_MAX_AGE_S = 24 * 3600


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def load_token(cfg):
    """Token ze souboru; při prvním spuštění se vygeneruje (práva jen pro vlastníka)."""
    path = Path(cfg["data_root"]) / "api-token"
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32) + "\n")
        log.info("vygenerován nový token: %s", path)
    return path.read_text().strip()


def upload_path(cfg, upload_id):
    if not UPLOAD_ID.match(upload_id or ""):
        raise ApiError(400, "neplatné upload_id")
    return ds.incoming_dir(cfg) / f"{upload_id}.flac"


def sidecar(path):
    return path.with_suffix(".json")


def cleanup_uploads(cfg):
    """Nahrávky, které aplikace nikdy neuložila ani nezahodila (spadla, ztratila síť)."""
    cutoff = time.time() - UPLOAD_MAX_AGE_S
    for p in ds.incoming_dir(cfg).glob("app_*.flac"):
        if p.stat().st_mtime < cutoff:
            remove_upload(p)


def run_cli(cfg, *args, timeout=120):
    """Příkaz voice-trainer → JSON ze stdout; hlášky stderr jdou do chyby."""
    cmd = [sys.executable, "-m", "voice_trainer.cli", *args]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cfg["repo_dir"])
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ApiError(500, err[-1] if err else f"voice-trainer {args[0]} selhal")
    out = proc.stdout.decode("utf-8", "replace").strip()
    return json.loads(out) if out else {}


def stt_reachable(cfg):
    parsed = urllib.parse.urlparse(cfg["stt_url"])
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=2):
            return True
    except OSError:
        return False


def audio_levels(path):
    """(rms_db, peak_db, délka) přes ffmpeg astats."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
         "astats=measure_overall=RMS_level+Peak_level:measure_perchannel=0",
         "-f", "null", "-"], capture_output=True, timeout=60)
    rms = peak = None
    for line in out.stderr.decode("utf-8", "replace").splitlines():
        for key, name in (("RMS level dB:", "rms"), ("Peak level dB:", "peak")):
            if key in line:
                try:
                    value = float(line.rsplit(":", 1)[1])
                except ValueError:
                    continue
                if name == "rms":
                    rms = value
                else:
                    peak = value
    info = ds.check_wav(path)
    return rms, peak, info[2] if info else None


def transcribe_upload(cfg, db, utterance_id, body):
    utt = db.utterance(utterance_id)
    if utt is None:
        raise ApiError(404, f"neznámá věta {utterance_id}")
    if not body:
        raise ApiError(400, "prázdná nahrávka")
    cleanup_uploads(cfg)  # server běží dlouho, úklid jen při startu by nestačil
    upload_id = "app_" + secrets.token_hex(12)
    dst = upload_path(cfg, upload_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(".part")
    tmp_dst.write_bytes(body)
    tmp_dst.replace(dst)
    with tempfile.TemporaryDirectory(prefix="vt-api-") as tmp:
        wav = Path(tmp) / "stt.wav"
        conv = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(dst),
             "-ar", str(cfg["recording"]["stt_sample_rate"]), "-ac", "1", str(wav)],
            capture_output=True, timeout=60)
        if conv.returncode != 0:
            dst.unlink(missing_ok=True)
            raise ApiError(400, "nahrávku nejde přečíst")
        rms, peak, duration = audio_levels(wav)
        try:
            text = stt_mod.transcribe(wav, cfg, utt["language"])
        except OSError as e:
            dst.unlink(missing_ok=True)
            raise ApiError(502, f"STT nedostupné: {e}") from e
    threshold = float(cfg["recording"]["similarity_threshold"])
    sim = stt_mod.similarity(utt["text"], text, utt["language"], cfg)
    # /save bere přepis odsud, ne z aplikace: rozhoduje, co slyšelo STT na serveru
    sidecar(dst).write_text(json.dumps({"utterance": utterance_id, "stt_text": text,
                                        "similarity": sim}, ensure_ascii=False))
    return {"upload_id": upload_id, "stt_text": text, "similarity": round(sim, 4),
            "threshold": threshold, "verdict": "PASS" if sim >= threshold else "FAIL",
            "rms_db": rms, "peak_db": peak, "duration": duration}


def remove_upload(path):
    path.unlink(missing_ok=True)
    sidecar(path).unlink(missing_ok=True)


def save_upload(cfg, data):
    session, utterance = str(data.get("session", "")), int(data.get("utterance", 0))
    src = upload_path(cfg, data.get("upload_id"))
    if not src.exists() or not sidecar(src).exists():
        raise ApiError(404, "nahrávka už na serveru není, nahraj ji znovu")
    checked = json.loads(sidecar(src).read_text())
    if checked["utterance"] != utterance:
        raise ApiError(409, "nahrávka patří k jiné větě")
    sim = checked["similarity"]
    if sim < float(cfg["recording"]["similarity_threshold"]):
        raise ApiError(409, f"věta neprošla ({sim * 100:.1f} %), do datasetu ji nejde uložit")
    result = run_cli(cfg, "save", "--session", session, "--utterance", str(utterance),
                     "--file", str(src), "--stt-text", checked["stt_text"],
                     "--similarity", f"{sim:.4f}", "--recorded-at", dbmod.now())
    remove_upload(src)
    return result


def list_recordings(db, query):
    name = query.get("speaker", [""])[0]
    sid = db.speaker_id(name)
    if sid is None:
        raise ApiError(404, f"neznámý mluvčí {name}")
    offset = max(0, int(query.get("offset", ["0"])[0]))
    limit = min(200, max(1, int(query.get("limit", ["50"])[0])))
    where = "r.speaker_id=? AND r.status IN ('PASS','REVIEW')"
    args = [sid]
    text = query.get("q", [""])[0].strip()
    if text:
        where += " AND u.text LIKE ?"
        args.append(f"%{text}%")
    total = db.one(f"SELECT COUNT(*) n FROM recordings r JOIN utterances u "
                   f"ON u.id = r.utterance_id WHERE {where}", args)["n"]
    rows = db.all(
        f"SELECT r.id, r.utterance_id, u.text, r.status, r.stt_text, r.similarity, "
        f"r.duration, COALESCE(r.recorded_at, r.checked_at) recorded_at, r.note "
        f"FROM recordings r JOIN utterances u ON u.id = r.utterance_id "
        f"WHERE {where} ORDER BY r.id DESC LIMIT ? OFFSET ?", [*args, limit, offset])
    return {"total": total, "offset": offset, "recordings": [dict(r) for r in rows]}


def recording_for(db, rid):
    rec = db.recording(rid)
    if rec is None or rec["status"] not in ("PASS", "REVIEW"):
        raise ApiError(404, f"nahrávka {rid} neexistuje")
    return rec


def recording_audio(cfg, db, rid):
    """Master (48 kHz) jako FLAC: přes ZeroTier je WAV zbytečně pomalý."""
    rec = recording_for(db, rid)
    sp = db.speaker(rec["speaker_id"])
    name = ds.utterance_name(rec["utterance_id"])
    path = next((p for p in (ds.master_dir(cfg, sp["name"]) / name,
                             ds.wav_dir(cfg, sp["name"]) / name) if p.exists()), None)
    if path is None:
        raise ApiError(404, "soubor nahrávky chybí")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-c:a", "flac", "-f", "flac", "-"], capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise ApiError(500, "nahrávku nejde převést")
    return proc.stdout


def delete_recording(cfg, db, rid):
    """Soubory do dataset/trash (lze ručně obnovit), v DB status DELETED. Stejnou větu
    a mluvčího můžou mít i starší řádky (retake sdílí soubor), proto všechny."""
    rec = recording_for(db, rid)
    sp = db.speaker(rec["speaker_id"])
    name = ds.utterance_name(rec["utterance_id"])
    trash = Path(cfg["data_root"]) / "dataset" / "trash" / sp["name"]
    trash.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = Path(name).stem
    for kind, folder in (("master", ds.master_dir(cfg, sp["name"])),
                         ("wav", ds.wav_dir(cfg, sp["name"]))):
        src = folder / name
        if src.exists():
            shutil.move(str(src), str(trash / f"{stem}.{stamp}.{kind}.wav"))
    # i SKIP: smazaná věta se musí nahrát znovu, přeskočení by ji z fronty dál drželo venku
    db.q("UPDATE recordings SET status='DELETED', note=? WHERE utterance_id=? AND "
         "speaker_id=? AND status IN ('PASS','REVIEW','SKIP')",
         (f"smazáno z aplikace {stamp}", rec["utterance_id"], rec["speaker_id"]))
    ds.write_metadata(db, cfg, rec["speaker_id"])
    log.info("smazána nahrávka %s (věta %s, mluvčí %s)", rid, rec["utterance_id"], sp["name"])
    return {"deleted": rid, "utterance_id": rec["utterance_id"]}


def checked_language(cfg, data):
    code = str(data.get("language") or languages.DEFAULT_LANGUAGE)
    if code not in languages.available(cfg):
        raise ApiError(400, f"neznámý jazyk {code}")
    return code


def set_language(cfg, db, name, data):
    sid = db.speaker_id(name)
    if sid is None:
        raise ApiError(404, f"neznámý mluvčí {name}")
    code = checked_language(cfg, data)
    if db.speaker(sid)["language"] != code and db.speaker_has_data(sid):
        raise ApiError(409, "hlas už má nahrávky, jazyk nejde změnit; pro jiný jazyk vytvoř nový hlas")
    return run_cli(cfg, "speaker-language", "--name", name, "--language", code)


def make_handler(cfg, token):
    class Handler(BaseHTTPRequestHandler):
        server_version = "voice-trainer-api"
        protocol_version = "HTTP/1.1"
        timeout = 700  # next-sentence může čekat na generování vět až 660 s

        def log_message(self, fmt, *args):
            log.info("%s %s", self.address_string(), fmt % args)

        def _send(self, status, body, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def _body(self):
            size = int(self.headers.get("Content-Length") or 0)
            if size > MAX_UPLOAD:
                self.close_connection = True
                raise ApiError(413, "nahrávka je moc velká")
            return self.rfile.read(size) if size else b""

        def _data(self):
            try:
                return json.loads(self._body() or b"{}")
            except ValueError as e:
                raise ApiError(400, "neplatný JSON") from e

        def _authorized(self):
            given = self.headers.get("Authorization", "")
            return hmac.compare_digest(given.encode(), f"Bearer {token}".encode())

        def _route(self, method):
            if not self._authorized():
                self._body()  # dočíst tělo, ať se spojení dá použít dál
                raise ApiError(401, "špatný token")
            url = urllib.parse.urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            query = urllib.parse.parse_qs(url.query)
            db = dbmod.Db(Path(cfg["data_root"]) / "voice-trainer.db")
            try:
                return self._dispatch(method, parts, query, db)
            finally:
                db.conn.close()

        def _dispatch(self, method, parts, query, db):
            route = (method, parts[0] if parts else "")
            if route == ("GET", "health"):
                return {"ok": True, "host": socket.gethostname(), "time": dbmod.now(),
                        "stt": stt_reachable(cfg),
                        "training_running": tr.training_running(cfg)}
            if route == ("GET", "speakers"):
                return run_cli(cfg, "speakers")
            if route == ("GET", "languages"):
                return run_cli(cfg, "languages")
            if route == ("POST", "speakers") and len(parts) == 3 and parts[2] == "language":
                return set_language(cfg, db, urllib.parse.unquote(parts[1]), self._data())
            if route == ("POST", "speakers") and len(parts) == 1:
                data = self._data()
                name = str(data.get("name", "")).strip()
                if not SPEAKER_NAME.match(name):
                    raise ApiError(400, "jméno smí mít jen písmena, číslice, mezeru a pomlčku")
                return run_cli(cfg, "new-speaker", "--name", name,
                               "--language", checked_language(cfg, data))
            if route == ("GET", "next"):
                speaker = query.get("speaker", [""])[0]
                if db.speaker_id(speaker) is None:
                    raise ApiError(404, f"neznámý mluvčí {speaker}")
                return run_cli(cfg, "next-sentence", "--speaker", speaker, timeout=660)
            if route == ("POST", "transcribe"):
                return transcribe_upload(cfg, db, int(query.get("utterance", ["0"])[0]),
                                         self._body())
            if route == ("POST", "save"):
                return save_upload(cfg, self._data())
            if route == ("POST", "discard"):
                remove_upload(upload_path(cfg, self._data().get("upload_id")))
                return {"ok": True}
            if route == ("POST", "skip"):
                data = self._data()
                run_cli(cfg, "skip", "--session", str(data.get("session", "")),
                        "--utterance", str(int(data.get("utterance", 0))),
                        "--note", "přeskočeno v aplikaci")
                return {"ok": True}
            if route == ("GET", "recordings") and len(parts) == 1:
                return list_recordings(db, query)
            if route == ("GET", "recordings") and len(parts) == 3 and parts[2] == "audio":
                return recording_audio(cfg, db, int(parts[1]))
            if route == ("DELETE", "recordings") and len(parts) == 2:
                return delete_recording(cfg, db, int(parts[1]))
            raise ApiError(404, "neznámá adresa")

        def _handle(self, method):
            try:
                result = self._route(method)
                if isinstance(result, bytes):
                    self._send(200, result, "audio/flac")
                else:
                    self._json(200, result)
            except ApiError as e:
                self._json(e.status, {"error": e.message})
            except ValueError as e:
                self._json(400, {"error": f"neplatná hodnota: {e}"})
            except subprocess.TimeoutExpired:
                self._json(504, {"error": "server nestihl odpovědět"})
            except BrokenPipeError:
                self.close_connection = True
            except Exception as e:  # odpověď i při neznámé chybě, jinak aplikace visí
                log.exception("chyba %s %s", method, self.path)
                self._json(500, {"error": str(e)})

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

        def do_DELETE(self):
            self._handle("DELETE")

    return Handler


def serve(cfg):
    token = load_token(cfg)
    host, port = cfg["api_host"], int(cfg["api_port"])
    server = ThreadingHTTPServer((host, port), make_handler(cfg, token))
    log.info("API běží na http://%s:%s", host, port)
    server.serve_forever()
