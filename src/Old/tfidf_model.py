
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_perfumes
from src.preprocessing import preprocess_text


def build_tfidf_model():
    perfumes = load_perfumes()

    # Preprocess every perfume description
    documents = perfumes["search_text"].apply(
        preprocess_text
    ).tolist()

    # Convert text into TF-IDF vectors
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    perfume_vectors = vectorizer.fit_transform(
        documents
    )

    return perfumes, vectorizer, perfume_vectors


def recommend_tfidf(query: str, top_k: int = 5):
    perfumes, vectorizer, perfume_vectors = (
        build_tfidf_model()
    )

    # Preprocess the user's query
    clean_query = preprocess_text(query)

    # Transform query using the fitted vocabulary
    query_vector = vectorizer.transform(
        [clean_query]
    )

    # Compare query with every perfume
    scores = cosine_similarity(
        query_vector,
        perfume_vectors
    ).flatten()

    # Rank highest similarity first
   # Keep only perfumes with a positive similarity score
    matched_indices = [
    i for i in scores.argsort()[::-1]
    if scores[i] > 0
]

    ranked_indices = matched_indices[:top_k]

    results = perfumes.iloc[ranked_indices].copy()
    results["similarity_score"] = scores[ranked_indices]
    return results[
        [
            "name",
            "description",
            "price_inr",
            "search_text",
            "similarity_score",
        ]
    ].reset_index(drop=True)


if __name__ == "__main__":
    query = "fresh woody perfume for summer"

    recommendations = recommend_tfidf(query)

    print("Query:", query)
    print(recommendations.to_string(index=False))
