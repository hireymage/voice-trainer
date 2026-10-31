"""Generování vět přes ollama."""
import json
import random
import re
import urllib.request

from voice_trainer import languages

def generate(cfg, count=None, language=languages.DEFAULT_LANGUAGE):
    """Vrátí seznam {count} vět v daném jazyce (čistý text, bez číslování)."""
    count = int(count or cfg["llm_batch_size"])
    lang = languages.get(cfg, language)
    # náhodný výběr témat → každá dávka míří jinam (jinak model krouží)
    topics = random.sample(lang["topics"], min(6, len(lang["topics"])))
    payload = {
        "model": cfg["llm_model"],
        "messages": [{"role": "user",
                      "content": lang["prompt"].format(count=count, language=lang["english_name"],
                                                       topics=", ".join(topics))}],
        "stream": False,
        "options": {"temperature": 0.9},
    }
    req = urllib.request.Request(
        cfg["llm_url"].rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=cfg["llm_timeout"]) as resp:
        body = json.load(resp)
    text = body.get("message", {}).get("content", "")

    out = []
    for line in text.splitlines():
        line = line.strip()
        # formát NNN|věta; tolerujte "1." i "1) " prefixy
        m = re.match(r"^(?:\d+[.)]?\s*\||\d+[.)]\s*)(.+)$", line)
        sent = (m.group(1) if m else line).strip().strip('"“”„').strip()
        if len(sent) >= 8 and re.search(r"[.!?]$", sent):
            out.append(sent)
        if len(out) >= count:
            break
    return out