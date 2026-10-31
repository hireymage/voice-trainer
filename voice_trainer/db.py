"""SQLite perzistence — dataset přežije restart i přerušené nahrávání.

Profily mluvčích (2026-09-26): nahrávky patří mluvčímu (recordings.speaker_id,
sessions.speaker_id); base model se rozhoduje až při tréninku (runs.speaker_id
+ runs.base_name = checkpoint chain per dvojice mluvčí × base).
"""
import datetime
import sqlite3
from pathlib import Path

# default speaker pro migraci starších dat (všechny dosavadní nahrávky = hozzy)
DEFAULT_SPEAKER = "hozzy"

SCHEMA = """
CREATE TABLE IF NOT EXISTS utterances (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL UNIQUE,
    source      TEXT NOT NULL DEFAULT 'llm',   -- llm | import | manual
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS speakers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,              -- session_2026_09_25_001
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    mic         TEXT,
    sample_rate INTEGER,
    batch_size  INTEGER,
    status      TEXT NOT NULL DEFAULT 'active', -- active | done
    speaker_id  INTEGER REFERENCES speakers(id)
);
CREATE TABLE IF NOT EXISTS recordings (
    id           INTEGER PRIMARY KEY,
    utterance_id INTEGER NOT NULL REFERENCES utterances(id),
    session_id   TEXT REFERENCES sessions(id),
    wav          TEXT,                          -- cesta vůči data_root/dataset/wav/<mluvčí>/
    master_wav   TEXT,                          -- raw 48k master (volitelné)
    status       TEXT NOT NULL DEFAULT 'PENDING',
                 -- PENDING | PASS | FAIL | RETAKE | SKIP | REVIEW | DELETED
    stt_text     TEXT,
    similarity   REAL,
    duration     REAL,
    sample_rate  INTEGER,
    channels     INTEGER,
    recorded_at  TEXT,
    checked_at   TEXT,
    note         TEXT,
    speaker_id   INTEGER REFERENCES speakers(id)
);
CREATE INDEX IF NOT EXISTS idx_rec_utt ON recordings(utterance_id);
CREATE INDEX IF NOT EXISTS idx_rec_sess ON recordings(session_id);
CREATE INDEX IF NOT EXISTS idx_rec_status ON recordings(status);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,          -- run_001
    dataset_dir TEXT,
    base_ckpt   TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
                -- running | done | failed | stopped
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    notes       TEXT,
    speaker_id  INTEGER REFERENCES speakers(id),
    base_name   TEXT                           -- label base modelu (jirka | kasandra | …)
);
"""

# indexy se tvoří AŽ PO _migrate() (idx_* na speaker_id by na legacy tabulce
# bez sloupce selhalo)
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_rec_utt ON recordings(utterance_id);
CREATE INDEX IF NOT EXISTS idx_rec_sess ON recordings(session_id);
CREATE INDEX IF NOT EXISTS idx_rec_status ON recordings(status);
CREATE INDEX IF NOT EXISTS idx_rec_speaker ON recordings(speaker_id);
"""


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class Db:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.executescript(INDEXES)
        self.conn.commit()

    def _migrate(self):
        """Idempotentní upgrade starší DB: sloupce speaker_id/base_name +
        backfill existujících dat do default speakeru (hozzy × jirka)."""
        def add_col(table, col, decl):
            cols = {r["name"] for r in self.conn.execute(
                f"PRAGMA table_info({table})")}
            if col not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        add_col("sessions", "speaker_id", "INTEGER REFERENCES speakers(id)")
        add_col("recordings", "speaker_id", "INTEGER REFERENCES speakers(id)")
        add_col("runs", "speaker_id", "INTEGER REFERENCES speakers(id)")
        add_col("runs", "base_name", "TEXT")
        # backfill: pokud už existují nahrávky/runy bez speakeru, jsou to
        # data default mluvčího
        has_rec = self.one("SELECT COUNT(*) n FROM recordings")["n"] > 0
        has_run = self.one("SELECT COUNT(*) n FROM runs")["n"] > 0
        if has_rec or has_run:
            sp = self.one("SELECT id FROM speakers WHERE name = ?",
                          (DEFAULT_SPEAKER,))
            if sp is None:
                self.q("INSERT INTO speakers (name, created_at) VALUES (?,?)",
                       (DEFAULT_SPEAKER, now()))
                sp = self.one("SELECT id FROM speakers WHERE name = ?",
                              (DEFAULT_SPEAKER,))
            sid = sp["id"]
            self.q("UPDATE recordings SET speaker_id=? WHERE speaker_id IS NULL",
                   (sid,))
            self.q("UPDATE sessions SET speaker_id=? WHERE speaker_id IS NULL",
                   (sid,))
            self.q("UPDATE runs SET speaker_id=? WHERE speaker_id IS NULL",
                   (sid,))
            # legacy runy řetězily jedinou bazi = jirka
            self.q("UPDATE runs SET base_name='jirka' WHERE base_name IS NULL")

    def q(self, sql, args=()):
        cur = self.conn.execute(sql, args)
        cur.fetchall()
        self.conn.commit()

    def one(self, sql, args=()):
        return self.conn.execute(sql, args).fetchone()

    def all(self, sql, args=()):
        return self.conn.execute(sql, args).fetchall()

    # ── speakers ─────────────────────────────────────────────────────────────
    def speakers(self):
        return self.all("SELECT * FROM speakers ORDER BY id")

    def speaker(self, sid):
        return self.one("SELECT * FROM speakers WHERE id = ?", (sid,))

    def speaker_id(self, name):
        row = self.one("SELECT id FROM speakers WHERE name = ?", (name,))
        return row["id"] if row else None

    def ensure_speaker(self, name):
        name = name.strip()
        if not name:
            raise ValueError("jméno mluvčího je prázdné")
        sid = self.speaker_id(name)
        if sid is None:
            self.q("INSERT INTO speakers (name, created_at) VALUES (?,?)",
                   (name, now()))
            sid = self.speaker_id(name)
        return sid

    def speaker_stats(self, sid):
        """(pass nahrávky, celková délka) pro mluvčího."""
        row = self.one(
            "SELECT COUNT(*) n, SUM(duration) d FROM recordings "
            "WHERE speaker_id=? AND status='PASS'", (sid,))
        return row["n"] or 0, row["d"] or 0.0

    # ── utterances ───────────────────────────────────────────────────────────
    def next_utterance_id(self):
        row = self.one("SELECT MAX(id) m FROM utterances")
        return (row["m"] or 0) + 1

    def add_utterance(self, text, source):
        text = text.strip()
        if not text:
            return None
        row = self.one("SELECT id FROM utterances WHERE text = ?", (text,))
        if row:  # duplicity textu = stejná věta, nevytvářet znovu
            return row["id"]
        uid = self.next_utterance_id()
        self.q("INSERT INTO utterances (id, text, source, created_at) VALUES (?,?,?,?)",
               (uid, text, source, now()))
        return uid

    def utterance(self, uid):
        return self.one("SELECT * FROM utterances WHERE id = ?", (uid,))

    def unrecorded_utterances(self, speaker_id=None):
        """Věty bez PASS, REVIEW ani SKIP nahrávky daného mluvčího (v pořadí ID).
        REVIEW se počítá jako vyřízené — nahrávka je uložená ke kontrole,
        smyčka pokračuje další větou (uživatel překlasifikuje přes review).
        SKIP taky, jinak by přeskočená věta přišla hned znovu."""
        if speaker_id is None:
            return self.all(
                "SELECT u.* FROM utterances u WHERE NOT EXISTS "
                "(SELECT 1 FROM recordings r WHERE r.utterance_id = u.id "
                " AND r.status IN ('PASS','REVIEW','SKIP')) "
                "ORDER BY u.id")
        return self.all(
            "SELECT u.* FROM utterances u WHERE NOT EXISTS "
            "(SELECT 1 FROM recordings r WHERE r.utterance_id = u.id "
            " AND r.status IN ('PASS','REVIEW','SKIP') AND r.speaker_id = ?) "
            "ORDER BY u.id", (speaker_id,))

    # ── sessions ─────────────────────────────────────────────────────────────
    def new_session_id(self):
        day = datetime.date.today().isoformat().replace("-", "_")
        n = 1
        for row in self.all("SELECT id FROM sessions WHERE id LIKE ?",
                            (f"session_{day}%",)):
            n = max(n, int(row["id"].rsplit("_", 1)[-1]) + 1)
        return f"session_{day}_{n:03d}"

    def create_session(self, batch_size, speaker_id=None):
        sid = self.new_session_id()
        self.q("INSERT INTO sessions (id, started_at, batch_size, status,"
               " speaker_id) VALUES (?,?,?,'active',?)",
               (sid, now(), batch_size, speaker_id))
        return sid

    def session(self, sid):
        return self.one("SELECT * FROM sessions WHERE id = ?", (sid,))

    def active_sessions(self, speaker_id=None):
        if speaker_id is None:
            return self.all("SELECT * FROM sessions WHERE status='active' "
                            "ORDER BY started_at")
        return self.all("SELECT * FROM sessions WHERE status='active' "
                        "AND speaker_id=? ORDER BY started_at", (speaker_id,))

    def finish_session(self, sid):
        self.q("UPDATE sessions SET status='done', ended_at=? WHERE id=?", (now(), sid))

    # ── recordings ───────────────────────────────────────────────────────────
    def add_recording(self, utterance_id, session_id, wav, status, stt_text,
                      similarity, duration, sample_rate, channels,
                      master_wav=None, note=None, recorded_at=None,
                      speaker_id=None):
        self.q(
            "INSERT INTO recordings (utterance_id, session_id, wav, master_wav,"
            " status, stt_text, similarity, duration, sample_rate, channels,"
            " recorded_at, checked_at, note, speaker_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (utterance_id, session_id, wav, master_wav, status, stt_text,
             similarity, duration, sample_rate, channels, recorded_at, now(),
             note, speaker_id))
        return self.one("SELECT * FROM recordings WHERE id = last_insert_rowid()")

    def update_recording_status(self, rid, status, note=None):
        self.q("UPDATE recordings SET status=?, note=? WHERE id=?", (status, note, rid))

    def recording(self, rid):
        return self.one("SELECT * FROM recordings WHERE id = ?", (rid,))

    def best_pass(self, utterance_id, speaker_id=None):
        """Poslední PASS nahrávka věty (per mluvčí)."""
        if speaker_id is None:
            return self.one(
                "SELECT * FROM recordings WHERE utterance_id=? AND status='PASS' "
                "ORDER BY id DESC LIMIT 1", (utterance_id,))
        return self.one(
            "SELECT * FROM recordings WHERE utterance_id=? AND status='PASS' "
            "AND speaker_id=? ORDER BY id DESC LIMIT 1",
            (utterance_id, speaker_id))

    def passes(self, speaker_id=None):
        """Nejnovější PASS per (věta, mluvčí) — pro metadata.csv."""
        if speaker_id is None:
            return self.all(
                "SELECT r.* FROM recordings r JOIN "
                "(SELECT utterance_id, MAX(id) mid FROM recordings WHERE status='PASS' "
                " GROUP BY utterance_id) g ON g.mid = r.id ORDER BY r.utterance_id")
        return self.all(
            "SELECT r.* FROM recordings r JOIN "
            "(SELECT utterance_id, MAX(id) mid FROM recordings WHERE status='PASS' "
            " AND speaker_id=? GROUP BY utterance_id) g ON g.mid = r.id "
            "ORDER BY r.utterance_id", (speaker_id,))

    def stats(self, speaker_id=None):
        if speaker_id is None:
            row = self.one(
                "SELECT COUNT(*) n, SUM(duration) d FROM recordings WHERE status='PASS'")
        else:
            row = self.one(
                "SELECT COUNT(*) n, SUM(duration) d FROM recordings "
                "WHERE status='PASS' AND speaker_id=?", (speaker_id,))
        return row["n"] or 0, row["d"] or 0.0

    # ── runs ─────────────────────────────────────────────────────────────────
    def new_run_name(self):
        n = 1
        for row in self.all("SELECT name FROM runs"):
            n = max(n, int(row["name"].rsplit("_", 1)[-1]) + 1)
        return f"run_{n:03d}"

    def create_run(self, name, dataset_dir, base_ckpt, speaker_id=None,
                   base_name=None):
        self.q("INSERT INTO runs (name, dataset_dir, base_ckpt, started_at,"
               " speaker_id, base_name) VALUES (?,?,?,?,?,?)",
               (name, dataset_dir, base_ckpt, now(), speaker_id, base_name))
        return self.one("SELECT * FROM runs WHERE name = ?", (name,))

    def update_run(self, rid, status, notes=None):
        self.q("UPDATE runs SET status=?, ended_at=?, notes=COALESCE(?, notes) WHERE id=?",
               (status, now(), notes, rid))

    def latest_run(self, speaker_id=None, base_name=None):
        sql, args = "SELECT * FROM runs", []
        where = []
        if speaker_id is not None:
            where.append("speaker_id=?"); args.append(speaker_id)
        if base_name is not None:
            where.append("base_name=?"); args.append(base_name)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT 1"
        return self.one(sql, args)