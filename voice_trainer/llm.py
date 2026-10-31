"""Generování vět přes ollama."""
import json
import random
import re
import urllib.request

# témata se losují na každou dávku — jinak model krouží pořád dokola kolem
# počasí/dopravy/nákupu (TTS dataset potřebuje hlavně fonetickou rozmanitost)
TOPICS = [
    "vaření a kuchyně", "domácí mazlíčci", "sport a cvičení", "cestování",
    "práce a kancelář", "zahrada a rostliny", "auta a doprava", "hudba",
    "knihy a čtení", "děti a škola", "nákup a peníze", "počasí a roční období",
    "zdraví a lékař", "technika a počítače", "filmy a seriály", "zahrádka",
    "rybaření a myslivost", "historie a památky", "kávovary a káva",
    "pronájmy a bydlení", "úklid a pořádek", "nástěnné hodiny a čas",
    "fakturace a účty", "zahraniční výlety", "cyklistika", "vlaky a stanice",
    "restaurace a jídlo venku", "nakupování online", "úklid bytu",
    "zimní sporty", "zahrada na jaře", "přátelé a rodina", "spánek a ranní rutina",
    "účetnictví a daně", "knihovna a čtení", "kino a divadlo",
]

PROMPT = (
    "Vygeneruj {count} krátkých českých vět pro nahrávání hlasového datasetu "
    "(TTS). Pravidla: přirozené běžné věty, 5 až 15 slov. TÉMATA PRO TENTO "
    "BATCH (střídej je mezi větami): {topics}. VĚTA NESMÍ OBSAHOVAT ČÍSLA, "
    "data ani roky — piš je slovy nebo je vynech (STT je pak píše číslicemi "
    "a kontrola shody nesedí). Každá věta na vlastní řádek ve formátu: "
    "NNN|věta. Žádný jiný text, žádné úvozovky."
)


def generate(cfg, count=None):
    """Vrátí seznam {count} českých vět (čistý text, bez číslování)."""
    count = int(count or cfg["llm_batch_size"])
    # náhodný výběr témat → každá dávka míří jinam (jinak model krouží)
    topics = random.sample(TOPICS, min(6, len(TOPICS)))
    payload = {
        "model": cfg["llm_model"],
        "messages": [{"role": "user",
                      "content": PROMPT.format(count=count,
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