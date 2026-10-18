
from src.data_loader import load_perfumes
from src.fragrance_features import (
    extract_accords,
    extract_notes,
    estimate_season,
    estimate_sweetness,
)


def main():
    perfumes = load_perfumes()

    for _, perfume in perfumes.head(10).iterrows():
        description = perfume["description"]

        print(f"\n{perfume['name']} — {perfume['brand']}")
        print("Accords:", extract_accords(description))
        print("Notes:", extract_notes(description))
        print("Season:", estimate_season(description))
        print("Sweetness:", estimate_sweetness(description))


if __name__ == "__main__":
    main()
