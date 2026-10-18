
import re


ACCORD_PATTERN = re.compile(
    r"Main accords:\s*(.*?)(?=\s*\|\s*Top notes:|$)",
    re.IGNORECASE,
)

NOTE_PATTERN = re.compile(
    r"(Top notes|Middle notes|Base notes):\s*(.*?)(?=\s*\|\s*(?:Top notes|Middle notes|Base notes):|$)",
    re.IGNORECASE,
)

SUMMER_TERMS = {
    "fresh", "citrus", "aquatic", "green",
    "marine", "bergamot", "lemon", "grapefruit",
    "lime", "mint", "neroli",
}

WINTER_TERMS = {
    "amber", "vanilla", "leather", "smoky",
    "resinous", "incense", "oud", "tobacco",
    "patchouli", "warm spicy",
}

SWEET_TERMS = {
    "sweet", "vanilla", "caramel", "honey",
    "chocolate", "gourmand", "tonka bean",
    "praline", "marshmallow",
}


def extract_accords(description: str) -> list[str]:
    if not isinstance(description, str):
        return []

    match = ACCORD_PATTERN.search(description)

    if not match:
        return []

    return [
        item.strip().lower()
        for item in match.group(1).split(",")
        if item.strip()
    ]


def extract_notes(description: str) -> list[str]:
    if not isinstance(description, str):
        return []

    notes = []

    for match in NOTE_PATTERN.finditer(description):
        notes.extend(
            note.strip().lower()
            for note in match.group(2).split(",")
            if note.strip()
        )

    return notes


def estimate_season(description: str) -> dict:
    accords = extract_accords(description)
    notes = extract_notes(description)

    # Use accords first; use notes only when accords are unavailable.
    features = set(accords if accords else notes)

    summer_score = sum(
        1 for term in SUMMER_TERMS
        if any(term == item or term in item for item in features)
    )

    winter_score = sum(
        1 for term in WINTER_TERMS
        if any(term == item or term in item for item in features)
    )

    if max(summer_score, winter_score) < 2:
        return {
            "season_estimate": None,
            "season_confidence": "insufficient evidence",
        }

    if summer_score >= winter_score + 2:
        season = "Summer"
    elif winter_score >= summer_score + 2:
        season = "Winter"
    else:
        return {
            "season_estimate": None,
            "season_confidence": "mixed signals",
        }

    return {
        "season_estimate": season,
        "season_confidence": "heuristic estimate",
    }



def estimate_sweetness(description: str) -> dict:
    """Experimental sweetness estimate from accords or notes."""
    accords = extract_accords(description)
    notes = extract_notes(description)

    features = set(accords if accords else notes)

    if not features:
        return {
            "sweetness_estimate": None,
            "sweetness_confidence": "insufficient evidence",
        }

    sweet_terms = {
        "sweet", "vanilla", "caramel", "honey",
        "chocolate", "gourmand", "tonka bean",
        "praline", "marshmallow", "sugar",
    }

    matches = {
        term
        for term in sweet_terms
        if any(term == item or term in item for item in features)
    }

    if "sweet" in features:
        score = 4
    elif len(matches) >= 3:
        score = 4
    elif len(matches) == 2:
        score = 3
    elif len(matches) == 1:
        score = 2
    else:
        score = None

    return {
        "sweetness_estimate": score,
        "sweetness_confidence": (
            "heuristic estimate" if score is not None
            else "insufficient evidence"
        ),
    }
