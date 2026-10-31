"""Jazyky hlasů. Nový jazyk je jedna položka tady nebo v config.yaml (languages:);
stačí mu název, ostatní má výchozí hodnotu podle kódu jazyka.

Jazyk patří k hlasu a k větě: určuje generování vět, kontrolu přes STT (whisper) i
fonémizaci při tréninku (espeak-ng)."""

DEFAULT_LANGUAGE = "cs"

# Pro jazyk bez vlastního zadání. Instrukce pro LLM je anglicky, věty píše v cílovém jazyce.
GENERIC_PROMPT = (
    "Generate {count} short sentences in {language} for recording a voice dataset "
    "(TTS). Rules: natural everyday sentences, 5 to 15 words. TOPICS FOR THIS BATCH "
    "(alternate between sentences): {topics}. A SENTENCE MUST NOT CONTAIN DIGITS, "
    "dates or years; spell them out or leave them out (STT writes them as digits "
    "and the match check fails). Each sentence on its own line in the format: "
    "NNN|sentence. No other text, no quotes."
)

GENERIC_TOPICS = [
    "cooking and kitchen", "pets", "sports and exercise", "travel", "work and office",
    "garden and plants", "cars and transport", "music", "books and reading",
    "children and school", "shopping and money", "weather and seasons",
    "health and doctors", "technology and computers", "films and series",
    "fishing and hunting", "history and monuments", "coffee", "renting and housing",
    "cleaning and tidying", "clocks and time", "invoices and bills", "trips abroad",
    "cycling", "trains and stations", "restaurants and eating out", "online shopping",
    "winter sports", "friends and family", "sleep and morning routine",
    "accounting and taxes", "cinema and theatre",
]

LANGUAGES = {
    "cs": {
        "name": "Čeština",
        "espeak_voice": "cs",
        # téma se losuje na každou dávku, jinak model krouží pořád kolem počasí/dopravy/nákupu
        # (TTS dataset potřebuje hlavně fonetickou rozmanitost)
        "topics": [
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
        ],
        "prompt": (
            "Vygeneruj {count} krátkých českých vět pro nahrávání hlasového datasetu "
            "(TTS). Pravidla: přirozené běžné věty, 5 až 15 slov. TÉMATA PRO TENTO "
            "BATCH (střídej je mezi větami): {topics}. VĚTA NESMÍ OBSAHOVAT ČÍSLA, "
            "data ani roky — piš je slovy nebo je vynech (STT je pak píše číslicemi "
            "a kontrola shody nesedí). Každá věta na vlastní řádek ve formátu: "
            "NNN|věta. Žádný jiný text, žádné úvozovky."
        ),
        # kmeny číslovek — tokeny začínající těmito kmeny (a číslice) se při srovnání
        # shovají: STT píše "12. srpna", referenční text "dvanáctého srpna", a takhle
        # by věta správně přečtená bezdůvodně FAILovala
        "number_stems": (
            "nula", "jedn", "dvě", "dvou", "dva", "tři", "tří", "čtyři", "čtyř",
            "pět", "šest", "sedm", "osm", "devět", "deset", "jedenáct", "dvanáct",
            "třináct", "čtrnáct", "patnáct", "šestnáct", "sedmnáct", "osmnáct",
            "devatenáct", "dvacet", "třicet", "čtyřicet", "padesát", "šedesát",
            "sedmdesát", "osmdesát", "devadesát", "sto", "tisíc",
            "první", "druh", "třetí", "čtvrt", "pát", "šest", "sedm", "osm", "devát",
            "desát",
        ),
    },
    "en": {
        "name": "English",
        "english_name": "English",
        "espeak_voice": "en-us",
        # celá slova: kmen "ten" by schoval i "tennis"
        "number_words": (
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
            "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
            "thousand", "million", "first", "second", "third", "fourth", "fifth",
            "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
        ),
    },
}


def _complete(code, entry):
    name = entry.get("name") or code
    return {
        "code": code,
        "name": name,
        "english_name": entry.get("english_name") or name,
        "espeak_voice": entry.get("espeak_voice") or code,
        "stt_language": entry.get("stt_language") or code,
        "prompt": entry.get("prompt") or GENERIC_PROMPT,
        "topics": list(entry.get("topics") or GENERIC_TOPICS),
        "number_stems": tuple(entry.get("number_stems") or ()),
        "number_words": frozenset(entry.get("number_words") or ()),
    }


def available(cfg=None):
    """Všechny jazyky: vestavěné + doplněné nebo upravené v config.yaml (languages:)."""
    merged = {code: dict(entry) for code, entry in LANGUAGES.items()}
    for code, entry in ((cfg or {}).get("languages") or {}).items():
        merged.setdefault(code, {}).update(entry or {})
    return {code: _complete(code, entry) for code, entry in merged.items()}


def get(cfg, code):
    """Jazyk podle kódu; neznámý kód je chyba, ať se dataset nezaplní větami v ničem."""
    langs = available(cfg)
    if code not in langs:
        raise ValueError(f"neznámý jazyk: {code} (dostupné: {', '.join(langs)})")
    return langs[code]
