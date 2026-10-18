
from pathlib import Path

import pandas as pd


# Find the project directory automatically
DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "perfumes.csv"
)


def load_perfumes():
    # Load CSV into a Pandas DataFrame
    df = pd.read_csv(DATA_PATH)

    # Columns required by our NLP pipeline
    required_columns = [
        "perfume_id",
        "name",
        "description",
        "top_notes",
        "middle_notes",
        "base_notes",
        "price_inr",
    ]

    # Check for missing columns
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Each perfume must have a unique ID
    if df["perfume_id"].duplicated().any():
        raise ValueError("Duplicate perfume IDs found.")

    # Clean missing descriptive text
    text_columns = [
        "description",
        "top_notes",
        "middle_notes",
        "base_notes",
    ]

    for column in text_columns:
        df[column] = df[column].fillna("").astype(str)

    # Combine fragrance information into one NLP text field
    df["search_text"] = (
        df["description"] + " "
        + df["top_notes"] + " "
        + df["middle_notes"] + " "
        + df["base_notes"]
    ).str.strip()

    return df


if __name__ == "__main__":
    perfumes = load_perfumes()

    print("Dataset loaded successfully!")
    print("Number of perfumes:", len(perfumes))
    print("Columns:", perfumes.columns.tolist())

    print("\nNLP search text:")
    print(
        perfumes[
            ["name", "search_text"]
        ].to_string(index=False)
    )
