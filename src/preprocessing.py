import re
def preprocess_text(text: str) -> str:
    if not isinstance(text, str): text = ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
