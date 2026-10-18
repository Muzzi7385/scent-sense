
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "parfumo_perfumes.csv.gz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "perfumes.csv"
)

TARGET_COUNT = 500

SOURCE_COLUMNS = [
    "Number",
    "Name",
    "Brand",
    "Main_Accords",
    "Top_Notes",
    "Middle_Notes",
    "Base_Notes",
]


def clean_text(value):
    """Convert missing values to empty strings and clean text."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_FILE}"
        )

    source = pd.read_csv(SOURCE_FILE)

    missing_columns = [
        col for col in SOURCE_COLUMNS
        if col not in source.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected source columns: {missing_columns}"
        )

    # Remove records without a usable name or brand.
    source["Name"] = source["Name"].map(clean_text)
    source["Brand"] = source["Brand"].map(clean_text)

    source = source[
        (source["Name"] != "")
        & (source["Brand"] != "")
    ].copy()

    # Use the source's perfume number to remove duplicate records.
    # Remove duplicate perfumes using their name and brand.
# Many records do not have a source Number, so Number alone
# cannot be used as a unique identifier.
    source["_dedupe_name"] = (
    source["Name"].str.casefold().str.strip()
)
    source["_dedupe_brand"] = (
    source["Brand"].str.casefold().str.strip()
)

    source = source.drop_duplicates(
    subset=["_dedupe_name", "_dedupe_brand"],
    keep="first",
    ).copy()

    # Prefer records with more fragrance information.
    note_columns = ["Top_Notes", "Middle_Notes", "Base_Notes"]
    source["_note_count"] = source[note_columns].notna().sum(axis=1)

    source = source.sort_values(
    by=["_note_count", "Name", "Brand"],
    ascending=[False, True, True],
    na_position="last",
)

    if len(source) < TARGET_COUNT:
        raise ValueError(
            f"Only {len(source)} usable perfumes found; "
            f"cannot create {TARGET_COUNT} records."
        )

    # Select 500 distinct source records.
    selected = source.head(TARGET_COUNT).copy()

    # Preserve source IDs while making the output IDs unique strings.
    # Generate reliable unique IDs independently of source Number.
    selected["perfume_id"] = [
    f"PFM_{i:04d}"
    for i in range(1, len(selected) + 1)
]

    selected["top_notes"] = selected["Top_Notes"].map(clean_text)
    selected["middle_notes"] = selected["Middle_Notes"].map(clean_text)
    selected["base_notes"] = selected["Base_Notes"].map(clean_text)

    selected["description"] = selected.apply(
        lambda row: " | ".join(
            part for part in [
                f"Main accords: {clean_text(row['Main_Accords'])}"
                if clean_text(row["Main_Accords"]) else "",
                f"Top notes: {clean_text(row['Top_Notes'])}"
                if clean_text(row["Top_Notes"]) else "",
                f"Middle notes: {clean_text(row['Middle_Notes'])}"
                if clean_text(row["Middle_Notes"]) else "",
                f"Base notes: {clean_text(row['Base_Notes'])}"
                if clean_text(row["Base_Notes"]) else "",
            ]
            if part
        ),
        axis=1,
    )

    # Keep the existing application schema.
    # Missing fields remain blank instead of being fabricated.
    selected["gender"] = ""
    selected["season"] = ""
    selected["occasion"] = ""
    selected["price_inr"] = pd.NA
    selected["sweetness"] = pd.NA
    selected["source"] = "Parfumo"

    output_columns = [
        "perfume_id",
        "name",
        "brand",
        "description",
        "top_notes",
        "middle_notes",
        "base_notes",
        "gender",
        "season",
        "occasion",
        "price_inr",
        "sweetness",
        "source",
    ]

    output = selected.rename(
        columns={
            "Name": "name",
            "Brand": "brand",
        }
    )[output_columns].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("Dataset conversion complete!")
    print(f"Source records available: {len(source)}")
    print(f"Records exported: {len(output)}")
    print(f"Unique perfume IDs: {output['perfume_id'].nunique()}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nMissing values by column:")
    print(output.isna().sum().to_string())
    print("\nSample records:")
    print(output.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
