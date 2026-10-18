from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def load_perfumes(path=None):
    candidates = [Path(path).expanduser()] if path else [
        ROOT/"data/processed/perfumes_catalogue.csv",
        ROOT/"data/processed/perfumes.csv",
        ROOT/"data/raw/perfumes.csv",
        ROOT/"data/raw/perfumes_backup.csv",
        ROOT/"data/perfumes_catalogue.csv",
        ROOT/"data/perfumes.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("Catalogue CSV not found. Keep your existing data folder; see README.md for supported paths.")
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"perfume_name":"name","fragrance":"name","perfume":"name","brand_name":"brand","notes":"description","profile":"description","price":"price_inr","price (inr)":"price_inr"}
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
    for col in ["name","brand","description","top_notes","middle_notes","base_notes","gender","season","occasion","search_text","price_inr"]:
        if col not in df.columns: df[col] = pd.NA if col == "price_inr" else ""
    for col in ["name","brand","description","top_notes","middle_notes","base_notes"]:
        df[col] = df[col].fillna("").astype(str)
    if "perfume_id" not in df.columns: df["perfume_id"] = [f"PFM_{i:05d}" for i in range(1,len(df)+1)]
    generated = df[["name","brand","description","top_notes","middle_notes","base_notes"]].fillna("").astype(str).agg(" ".join,axis=1)
    existing = df["search_text"].fillna("").astype(str)
    df["search_text"] = existing.where(existing.str.strip().ne(""), generated).fillna("").astype(str)
    if df.empty: raise ValueError(f"Catalogue is empty: {csv_path}")
    return df.reset_index(drop=True)
