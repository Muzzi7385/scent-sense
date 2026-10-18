import re

import pandas as pd

from src.fragrance_features import estimate_season, estimate_sweetness


def extract_constraints(query):
    """Extract optional budget, season, and sweetness constraints."""
    q = str(query).lower()

    constraints = {
        "max_price": None,
        "season": None,
        "max_sweetness": None,
    }

    price_patterns = [
        r"(?:under|below|less than|maximum|max(?:imum)?|up to|upto)"
        r"\s*₹?\s*([\d,]+)",
        r"budget\s*(?:of|under|up to)?\s*₹?\s*([\d,]+)",
        r"within\s+(?:a\s+)?budget\s+of\s*₹?\s*([\d,]+)",
    ]

    for pattern in price_patterns:
        match = re.search(pattern, q)
        if match:
            constraints["max_price"] = float(
                match.group(1).replace(",", "")
            )
            break

    for season in ("summer", "winter", "spring", "autumn", "fall"):
        if re.search(rf"\b{season}\b", q):
            constraints["season"] = (
                "Autumn" if season == "fall" else season.title()
            )
            break

    if re.search(
        r"\b(not sweet|less sweet|non[- ]sweet|not too sweet|unsweetened)\b",
        q,
    ):
        constraints["max_sweetness"] = 2

    return constraints


def _estimate_value(function, description, key):
    """Safely extract an estimated feature from a perfume description."""
    try:
        result = function(str(description or ""))
        if isinstance(result, dict):
            return result.get(key)
        return None
    except (TypeError, ValueError, AttributeError):
        return None


def _get_feature_series(df, column, description_column, estimator, key):
    """Use existing estimates when available; otherwise estimate from text."""
    if column in df.columns:
        values = df[column].copy()
    else:
        values = pd.Series(index=df.index, dtype="object")

    if description_column in df.columns:
        missing = values.isna() | values.astype(str).str.strip().eq("")
        if missing.any():
            values.loc[missing] = df.loc[
                missing, description_column
            ].apply(
                lambda description: _estimate_value(
                    estimator, description, key
                )
            )

    return values.fillna("").astype(str).str.strip().str.lower()


def apply_constraints(df, constraints):
    """Filter a perfume catalogue using extracted constraints."""
    out = df.copy()

    max_price = constraints.get("max_price")
    season = constraints.get("season")
    max_sweetness = constraints.get("max_sweetness")

    if max_price is not None:
        if "price_inr" not in out.columns:
            return out.iloc[0:0].reset_index(drop=True)

        prices = pd.to_numeric(out["price_inr"], errors="coerce")
        out = out.loc[prices.notna() & (prices <= max_price)].copy()

    if season is not None:
        season_values = _get_feature_series(
            out,
            "season_estimate",
            "description",
            estimate_season,
            "season_estimate",
        )

        # Accept common combined-season labels.
        desired = str(season).strip().lower()
        allowed = {
            "spring": {"spring", "spring/summer"},
            "summer": {"summer", "spring/summer"},
            "autumn": {"autumn", "autumn/winter", "fall"},
            "winter": {"winter", "autumn/winter"},
        }.get(desired, {desired})

        out = out.loc[season_values.isin(allowed)].copy()

    if max_sweetness is not None:
        sweetness_values = _get_feature_series(
            out,
            "sweetness_estimate",
            "description",
            estimate_sweetness,
            "sweetness_estimate",
        )

        sweetness_numbers = pd.to_numeric(
            sweetness_values, errors="coerce"
        )
        out = out.loc[
            sweetness_numbers.notna()
            & (sweetness_numbers <= max_sweetness)
        ].copy()

    return out.reset_index(drop=True)
