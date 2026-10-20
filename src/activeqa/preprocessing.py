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
