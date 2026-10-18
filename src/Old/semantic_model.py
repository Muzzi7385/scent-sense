
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_perfumes


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_semantic_model():
    perfumes = load_perfumes()

    # Load a pretrained sentence-embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Convert perfume descriptions into embeddings
    perfume_embeddings = model.encode(
        perfumes["search_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return perfumes, model, perfume_embeddings


def recommend_semantic(query: str, top_k: int = 5):
    perfumes, model, perfume_embeddings = (
        build_semantic_model()
    )

    # Convert the user's query into an embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    # Compare the query with all perfume embeddings
    scores = cosine_similarity(
        query_embedding,
        perfume_embeddings,
    ).flatten()

    # Rank perfumes by semantic similarity
    ranked_indices = scores.argsort()[::-1][:top_k]

    results = perfumes.iloc[ranked_indices].copy()
    results["semantic_score"] = scores[ranked_indices]

    return results[
        [
            "name",
            "description",
            "price_inr",
            "semantic_score",
        ]
    ].reset_index(drop=True)


if __name__ == "__main__":
    query = "Something fresh and woody for hot weather"

    recommendations = recommend_semantic(query)

    print("\nQuery:", query)
    print("\nSemantic recommendations:")
    print(recommendations.to_string(index=False))
