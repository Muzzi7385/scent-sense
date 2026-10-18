
import re


def preprocess_text(text: str) -> str:
    """
    Normalize text before NLP processing.
    """
    if not isinstance(text, str):
        text = ""

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation while preserving useful words
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
