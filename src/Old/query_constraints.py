
import re

from src.fragrance_features import (
    estimate_season,
    estimate_sweetness,
)


def extract_constraints(query: str) -> dict:
    text = query.lower()

    constraints = {
        "max_price": None,
        "season": None,
        "max_sweetness": None,
    }

    budget_match = re.search(
        r"(?:under|below|less than|maximum|max\.?|"
        r"within a budget of|budget of|budget under|"
        r"up to|upto)\s*(?:₹|rs\.?\s*)?([\d,]+)",
        text,
    )

    if budget_match:
        constraints["max_price"] = int(
            budget_match.group(1).replace(",", "")
        )

    if any(word in text for word in [
        "summer", "hot weather", "hot climate"
    ]):
        constraints["season"] = "Summer"
    elif any(word in text for word in [
        "winter", "cold weather", "cold climate"
    ]):
        constraints["season"] = "Winter"
    elif "spring" in text:
        constraints["season"] = "Spring"
    elif any(word in text for word in ["autumn", "fall"]):
        constraints["season"] = "Autumn"

    if any(phrase in text for phrase in [
        "not sweet", "non sweet", "unsweetened",
        "less sweet", "avoid sweet",
    ]):
        constraints["max_sweetness"] = 2

    return constraints


def apply_constraints(perfumes, constraints):
    filtered = perfumes.copy()

    if constraints["max_price"] is not None:
        filtered = filtered[
            filtered["price_inr"].notna()
            & (
                filtered["price_inr"]
                < constraints["max_price"]
            )
        ]

    if constraints["season"] is not None:
        estimates = filtered["description"].apply(
            estimate_season
        )

        estimated_seasons = estimates.apply(
            lambda result: result["season_estimate"]
        )

        filtered = filtered[
            estimated_seasons == constraints["season"]
        ].copy()

    if constraints["max_sweetness"] is not None:
        estimates = filtered["description"].apply(
            estimate_sweetness
        )

        estimated_sweetness = estimates.apply(
            lambda result: result["sweetness_estimate"]
        )

        filtered = filtered[
            estimated_sweetness.notna()
            & (
                estimated_sweetness
                <= constraints["max_sweetness"]
            )
        ].copy()

    return filtered
