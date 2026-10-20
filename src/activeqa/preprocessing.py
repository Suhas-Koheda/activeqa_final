import re, unicodedata

def normalize(text):
    text = unicodedata.normalize("NFC", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()

def tokens(text):
    # Python \w excludes many Indic combining marks; keep Unicode letters, numbers and marks.
    out=[]; current=[]
    for ch in normalize(text).casefold():
        if unicodedata.category(ch)[0] in {"L", "N", "M"}: current.append(ch)
        elif current:
            out.append("".join(current)); current=[]
    if current: out.append("".join(current))
    return out

def entity_signature(text):
    # Conservative protected spans: numerals and capitalized Latin names. Telugu proper names
    # cannot be safely detected without language-specific NER, so preservation is sampled/audited.
    return set(re.findall(r"\d+(?:[./:-]\d+)*|\b[A-Z][\w-]*\b", text))


# Telugu interrogative words grouped by answer type. A paraphrase that turns
# "which tradition" into "why do they follow it" changes the answer type even
# when all names and numbers survive, so this is screened separately.
_INTERROGATIVES = {
    "ఏ": "which", "ఏది": "which",
    "ఏమి": "what", "ఏమిటి": "what",
    "ఎవరు": "who", "ఎవరిని": "who", "ఎవరికి": "who", "ఎవరి": "who",
    "ఎవరిది": "who",
    "ఎప్పుడు": "when", "ఎప్పటి": "when", "ఎప్పటికి": "when",
    "ఎప్పటిది": "when",
    "ఎక్కడ": "where", "ఎక్కడి": "where", "ఎక్కడికి": "where",
    "ఎక్కడిది": "where",
    "ఎందుకు": "why", "ఎందుకని": "why",
    "ఎలా": "how", "ఎలాంటి": "how", "ఎలాంటిది": "how",
    "ఎంత": "how_much", "ఎంతకు": "how_much", "ఎంతమంది": "how_many",
    "ఎన్ని": "how_many",
}


def interrogatives(text):
    """Return the answer-type categories of the Telugu interrogatives in text."""
    return {_INTERROGATIVES[token] for token in tokens(text) if token in _INTERROGATIVES}
