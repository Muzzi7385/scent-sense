import re

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.data_loader import load_perfumes
from src.preprocessing import preprocess_text
from src.tfidf_model import build_tfidf_model
from src.semantic_model import build_semantic_model
from src.fragrance_features import (
    estimate_season,
    estimate_sweetness,
)
from src.query_constraints import (
    extract_constraints,
    apply_constraints,
)


# Translate descriptive moods into possible fragrance characteristics.
# These are experimental recommendations, not universal fragrance rules.
VIBE_PROFILES = {
    "seductive": [
        "woody", "spicy", "amber", "musk", "sweet"
    ],
    "masculine": [
        "woody", "spicy", "leather", "oud", "musk"
    ],
    "mysterious": [
        "woody", "smoky", "amber", "oud", "spicy"
    ],
    "elegant": [
        "floral", "woody", "musk", "amber"
    ],
    "romantic": [
        "floral", "sweet", "vanilla", "musk"
    ],
    "fresh": [
        "fresh", "citrus", "green", "aquatic"
    ],
    "clean": [
        "fresh", "aquatic", "citrus", "musk"
    ],
    "energetic": [
        "citrus", "fresh", "spicy", "fruity"
    ],
}


# Characteristics used to compare a request with a perfume.
FEATURE_GROUPS = {
    "fresh": ["fresh", "clean", "crisp"],
    "woody": ["woody", "wood", "cedarwood", "sandalwood", "oud"],
    "citrus": ["citrus", "lemon", "bergamot", "grapefruit", "orange"],
    "green": ["green", "grassy", "herbal"],
    "floral": ["floral", "flower", "rose", "jasmine"],
    "sweet": ["sweet", "gourmand", "vanilla", "caramel", "honey"],
    "spicy": ["spicy", "pepper", "cinnamon", "nutmeg", "saffron"],
    "aquatic": ["aquatic", "marine", "oceanic"],
    "fruity": ["fruity", "fruit", "berry", "apple", "peach"],
    "smoky": ["smoky", "smoke", "incense"],
    "vanilla": ["vanilla", "vanillin"],
    "musk": ["musk", "musky", "muskiness"],
    "amber": ["amber", "ambery"],
    "leather": ["leather", "leathery"],
    "oud": ["oud", "agarwood"],
}


def expand_vibe_query(query: str) -> str:
    """
    Convert mood and impression words into fragrance characteristics.

    Example:
    'seductive and masculine'
    becomes a query containing woody, spicy, amber, musk,
    sweet, leather and oud characteristics.
    """
    query_lower = query.lower()
    added_features = []

    for vibe, features in VIBE_PROFILES.items():
        pattern = rf"\b{re.escape(vibe)}\b"

        if re.search(pattern, query_lower):
            added_features.extend(features)

    # Remove duplicate characteristics while preserving their order.
    added_features = list(dict.fromkeys(added_features))

    if not added_features:
        return query

    return query + " " + " ".join(added_features)


def calculate_feature_match(query: str, description: str):
    """
    Calculate how many requested fragrance characteristics
    appear in a perfume's text.

    Returns:
        feature_match_score: value between 0 and 1
        matched_features: readable list of matched characteristics
    """
    query_tokens = set(re.findall(r"[a-z]+", str(query).lower()))
    description_tokens = set(
        re.findall(r"[a-z]+", str(description).lower())
    )

    requested_features = []

    for feature, synonyms in FEATURE_GROUPS.items():
        if any(synonym in query_tokens for synonym in synonyms):
            requested_features.append(feature)

    if not requested_features:
        return 0.0, "No direct feature match"

    matched_features = []

    for feature in requested_features:
        synonyms = FEATURE_GROUPS[feature]

        if any(synonym in description_tokens for synonym in synonyms):
            matched_features.append(feature)

    score = len(matched_features) / len(requested_features)

    if matched_features:
        readable_features = ", ".join(matched_features)
    else:
        readable_features = "No direct feature match"

    return score, readable_features


def normalize_scores(scores):
    """Scale model scores to a comparable range between 0 and 1."""
    scores_min = scores.min()
    scores_max = scores.max()

    if scores_max == scores_min:
        return scores * 0

    return (scores - scores_min) / (scores_max - scores_min)


def get_estimate_value(result):
    """
    Support feature estimators that return either a single value
    or a tuple containing an estimate and additional information.
    """
    if isinstance(result, tuple):
        return result[0]

    return result


def recommend_hybrid(
    query: str,
    top_k: int = 5,
    alpha: float = 0.5,
):
    """
    Recommend perfumes using:
    1. TF-IDF text similarity
    2. Semantic similarity
    3. Direct fragrance-characteristic matching
    4. Existing season, sweetness and budget constraints

    alpha controls the balance between TF-IDF and semantic similarity.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please enter a fragrance description.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    # Preserve the original query for budget and other constraints.
    constraints = extract_constraints(query)

    # Expand mood words such as seductive, elegant and mysterious.
    search_query = expand_vibe_query(query)
    clean_query = preprocess_text(search_query)

    # ---------------------------------------------------------
    # 1. TF-IDF similarity
    # ---------------------------------------------------------
    perfumes, vectorizer, perfume_vectors = build_tfidf_model()

    query_vector = vectorizer.transform([clean_query])

    tfidf_scores = cosine_similarity(
        query_vector,
        perfume_vectors,
    ).flatten()

    # ---------------------------------------------------------
    # 2. Semantic similarity
    # ---------------------------------------------------------
    semantic_perfumes, model, perfume_embeddings = (
        build_semantic_model()
    )

    query_embedding = model.encode(
        [search_query],
        normalize_embeddings=True,
    )

    raw_semantic_scores = cosine_similarity(
        query_embedding,
        perfume_embeddings,
    ).flatten()

    # Match semantic scores to the TF-IDF catalogue by perfume ID.
    semantic_score_lookup = dict(
        zip(
            semantic_perfumes["perfume_id"],
            raw_semantic_scores,
        )
    )

    semantic_scores = (
        perfumes["perfume_id"]
        .map(semantic_score_lookup)
        .fillna(0)
        .to_numpy()
    )

    # ---------------------------------------------------------
    # 3. Normalize and combine the model scores
    # ---------------------------------------------------------
    tfidf_normalized = normalize_scores(tfidf_scores)
    semantic_normalized = normalize_scores(semantic_scores)

    hybrid_scores = (
        alpha * tfidf_normalized
        + (1 - alpha) * semantic_normalized
    )

    # ---------------------------------------------------------
    # 4. Match requested fragrance characteristics
    # ---------------------------------------------------------
    feature_results = perfumes.apply(
        lambda row: calculate_feature_match(
            search_query,
            row.get("search_text", row.get("description", "")),
        ),
        axis=1,
    )

    feature_scores = feature_results.apply(lambda result: result[0])
    matched_features = feature_results.apply(lambda result: result[1])

    # ---------------------------------------------------------
    # 5. Estimate season and sweetness
    # ---------------------------------------------------------
    scored = perfumes.copy()

    scored["season_estimate"] = scored["description"].apply(
        lambda description: get_estimate_value(
            estimate_season(description)
        )
    )

    scored["sweetness_estimate"] = scored["description"].apply(
        lambda description: get_estimate_value(
            estimate_sweetness(description)
        )
    )

    # ---------------------------------------------------------
    # 6. Calculate final ranking
    # ---------------------------------------------------------
    scored["tfidf_score"] = tfidf_scores
    scored["semantic_score"] = semantic_scores
    scored["hybrid_score"] = hybrid_scores
    scored["feature_match_score"] = feature_scores
    scored["matched_features"] = matched_features

    scored["final_score"] = (
        0.7 * scored["hybrid_score"]
        + 0.3 * scored["feature_match_score"]
    )

    # Apply constraints from the ORIGINAL user query.
    filtered = apply_constraints(scored, constraints)

    if filtered.empty:
        return filtered

    # Highest-ranked results first.
    filtered = filtered.sort_values(
        by="final_score",
        ascending=False,
    )

    result_columns = [
        "name",
        "brand",
        "description",
        "price_inr",
        "season_estimate",
        "sweetness_estimate",
        "tfidf_score",
        "semantic_score",
        "hybrid_score",
        "feature_match_score",
        "matched_features",
        "final_score",
    ]

    available_columns = [
        column
        for column in result_columns
        if column in filtered.columns
    ]

    return filtered[available_columns].head(top_k).reset_index(drop=True)


if __name__ == "__main__":
    query = (
        "I have a date tonight. I want something seductive, "
        "masculine, and memorable for a first impression."
    )

    print("\n🌸 SCENT SENSE — AI FRAGRANCE CONSULTANT")
    print("=" * 65)
    print(f"Your request: {query}")

    print("\nUnderstanding your fragrance preferences...")
    print(f"Expanded fragrance profile: {expand_vibe_query(query)}")

    print("\nFinding your recommendations...\n")

    try:
        recommendations = recommend_hybrid(query, top_k=5)

        if recommendations.empty:
            print(
                "No perfumes matched all the current constraints. "
                "Try a broader description."
            )
        else:
            for index, perfume in recommendations.iterrows():
                print(f"{index + 1}. {perfume['name']}")
                print(f"   Brand: {perfume['brand']}")
                print(
                    "   Matching characteristics: "
                    f"{perfume['matched_features']}"
                )
                print(
                    "   Estimated season: "
                    f"{perfume['season_estimate']}"
                )
                print(
                    "   Estimated sweetness: "
                    f"{perfume['sweetness_estimate']}"
                )
                print(
                    "   Ranking score: "
                    f"{perfume['final_score']:.3f}"
                )
                print("-" * 65)

    except Exception as error:
        print(f"Something went wrong: {error}")
        raise
