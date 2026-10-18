
from src.data_loader import load_perfumes
from src.query_constraints import (
    extract_constraints,
    apply_constraints,
)


def test_constraint(query):
    perfumes = load_perfumes()
    constraints = extract_constraints(query)
    results = apply_constraints(perfumes, constraints)

    print(f"\nQuery: {query}")
    print(f"Constraints: {constraints}")
    print(f"Matching perfumes: {len(results)}")

    if constraints["max_price"] is not None:
        assert results["price_inr"].notna().all()
        assert (
            results["price_inr"] < constraints["max_price"]
        ).all()

    if constraints["season"] is not None:
        estimated = results["description"].apply(
            lambda d: __import__(
                "src.fragrance_features",
                fromlist=["estimate_season"],
            ).estimate_season(d)["season_estimate"]
        )
        assert (estimated == constraints["season"]).all()

    if constraints["max_sweetness"] is not None:
        from src.fragrance_features import estimate_sweetness

        estimated = results["description"].apply(
            lambda d: estimate_sweetness(d)["sweetness_estimate"]
        )
        assert estimated.notna().all()
        assert (estimated <= constraints["max_sweetness"]).all()

    print("PASS: No constraint violations.")

    if not results.empty:
        print(
            results[["name", "brand"]]
            .head(5)
            .to_string(index=False)
        )


if __name__ == "__main__":
    test_constraint("Perfume for summer")
    test_constraint("Perfume for winter")
    test_constraint("Perfume that is not sweet")
    test_constraint("Fresh perfume for summer under 3000")
