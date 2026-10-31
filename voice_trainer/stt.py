"""STT (whisper.cpp server) + podobnost textů.

Používá se na serveru pro `check` (opětovné ověření nahrávky). Klient
(voice-trainer-record na Macu) má stejnou logiku vestavěnou, aby nemusel
vždy obíhat přes server.
"""
import difflib
import json
import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path


def normalize(text):
    """Lowercase, bez interpunkce/diakritiky-rozdílů/whitespace → srovnatelné."""
    t = text.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\sčřžýáíéúůěšťďňó]", " ", t, flags=re.UNICODE)
    t = re.sub(r"[_\d]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# kmeny českých číslovek — tokeny začínající těmito kmeny (a číslice) se při
# srovnání shovají: STT píše "12. srpna", referenční text "dvanáctého srpna",
# a takhle by věta správně přečtená bezdůvodně FAILovala
NUM_STEMS = (
    "nula", "jedn", "dvě", "dvou", "dva", "tři", "tří", "čtyři", "čtyř",
    "pět", "šest", "sedm", "osm", "devět", "deset", "jedenáct", "dvanáct",
    "třináct", "čtrnáct", "patnáct", "šestnáct", "sedmnáct", "osmnáct",
    "devatenáct", "dvacet", "třicet", "čtyřicet", "padesát", "šedesát",
    "sedmdesát", "osmdesát", "devadesát", "sto", "tisíc",
    "první", "druh", "třetí", "čtvrt", "pát", "šest", "sedm", "osm", "devát",
    "desát",
)


def _no_numbers(text):
    """Odebere tokeny, které jsou číslicí nebo číslovkou (dle kmene)."""
    out = []
    for tok in text.split():
        if tok.isdigit() or any(tok.startswith(s) for s in NUM_STEMS):
            continue
        out.append(tok)
    return " ".join(out)


def similarity(reference, hypothesis):
    """Podíl shody 0.0–1.0 na normalizovaných textech (char-level).
    Bere se MAX ze srovnání plných textů a textů bez číselných tokenů —
    číslo zapsané slovy vs. číslicí tedy nikdy nevyvolá falešný FAIL."""
    a, b = normalize(reference), normalize(hypothesis)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    base = difflib.SequenceMatcher(None, a, b).ratio()
    na, nb = _no_numbers(a), _no_numbers(b)
    if not na or not nb:
        return base
    return max(base, difflib.SequenceMatcher(None, na, nb).ratio())


def cer(reference, hypothesis):
    """Character Error Rate na normalizovaných textech."""
    a, b = normalize(reference), normalize(hypothesis)
    if not a:
        return 0.0 if not b else 1.0
    dist = 0
    for tag, i0, i1, j0, j1 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "replace":
            dist += max(i1 - i0, j1 - j0)
        elif tag == "delete":
            dist += i1 - i0
        elif tag == "insert":
            dist += j1 - j0
    return dist / len(a)


def transcribe(wav_path, cfg):
    """Pošle WAV na whisper server, vrátí (text, None). Cold-start gate může
    blokovat spojení až ~120 s — proto timeout + retry."""
    for attempt in range(1 + int(cfg["stt_retries"])):
        try:
            with open(wav_path, "rb") as f:
                boundary = "----vtFormBoundary"
                parts = []
                for name, val in (("file", None), ("response_format", "text"),
                                  ("language", cfg["stt_language"])):
                    if val is None:  # file → multipart body níže
                        continue
                    parts.append(
                        f'--{boundary}\r\nContent-Disposition: form-data; '
                        f'name="{name}"\r\n\r\n{val}\r\n'.encode())
                filename = Path(wav_path).name
                parts.append(
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                    f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
                parts.append(f.read())
                parts.append(f"\r\n--{boundary}--\r\n".encode())
                body = b"".join(parts)
            req = urllib.request.Request(
                cfg["stt_url"], data=body, method="POST",
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            with urllib.request.urlopen(req, timeout=int(cfg["stt_timeout"])) as resp:
                text = resp.read().decode("utf-8", "replace")
            try:
                return json.loads(text).get("text", "").strip()
            except (json.JSONDecodeError, ValueError):
                return text.strip()
        except urllib.error.URLError as e:
            if attempt < int(cfg["stt_retries"]):
                continue
            raise
    return ""