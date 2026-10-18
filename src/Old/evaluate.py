
import math
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.data_loader import load_perfumes
from src.preprocessing import preprocess_text
from src.tfidf_model import build_tfidf_model
from src.semantic_model import MODEL_NAME
from src.hybrid_model import normalize_scores
from src.query_constraints import extract_constraints, apply_constraints


def precision_at_k(ranked_ids, relevant_ids, k):
    top_k = ranked_ids[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        item in relevant_ids for item in top_k
    )

    return relevant_count / len(top_k)


def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0

    top_k = ranked_ids[:k]

    return sum(item in relevant_ids for item in top_k) / len(relevant_ids)


def ndcg_at_k(ranked_ids, relevant_ids, k):
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, perfume_id in enumerate(ranked_ids[:k], start=1)
        if perfume_id in relevant_ids
    )

    ideal_count = min(len(relevant_ids), k)

    if ideal_count == 0:
        return 0.0

    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / ideal_dcg


def rank_scores(frame, score_column):
    return frame.sort_values(
        score_column, ascending=False
    )["perfume_id"].tolist()


def evaluate():
    queries = pd.read_csv("data/queries.csv")
    perfumes = load_perfumes()

    # Build each model once for this evaluation run.
    tfidf_perfumes, vectorizer, perfume_vectors = build_tfidf_model()

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        perfumes["search_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Match TF-IDF rows to the catalogue's perfume IDs.
    tfidf_perfumes = tfidf_perfumes.reset_index(drop=True)

    results = []

    for _, query_row in queries.iterrows():
        query = query_row["query"]
        relevant_ids = set(
            str(query_row["relevant_perfume_ids"]).split(";")
        )

        # Use identical explicit constraints for all models.
        constraints = extract_constraints(query)

        # TF-IDF scores
        query_vector = vectorizer.transform([
            preprocess_text(query)
        ])

        tfidf_scores = cosine_similarity(
            query_vector, perfume_vectors
        ).flatten()

        tfidf_frame = tfidf_perfumes.copy()
        tfidf_frame["tfidf_score"] = tfidf_scores

        # Semantic scores
        query_embedding = model.encode(
            [query], normalize_embeddings=True
        )

        semantic_scores = cosine_similarity(
            query_embedding, embeddings
        ).flatten()

        semantic_frame = perfumes.copy()
        semantic_frame["semantic_score"] = semantic_scores

        # Hybrid scores: same normalization and alpha as the prototype.
        hybrid_frame = perfumes.copy()
        hybrid_frame["tfidf_score"] = tfidf_scores
        hybrid_frame["semantic_score"] = semantic_scores

        hybrid_frame["tfidf_normalized"] = normalize_scores(
            hybrid_frame["tfidf_score"]
        )
        hybrid_frame["semantic_normalized"] = normalize_scores(
            hybrid_frame["semantic_score"]
        )

        alpha = 0.5
        hybrid_frame["hybrid_score"] = (
            alpha * hybrid_frame["tfidf_normalized"]
            + (1 - alpha) * hybrid_frame["semantic_normalized"]
        )

        model_frames = {
            "TF-IDF": (tfidf_frame, "tfidf_score"),
            "Semantic": (semantic_frame, "semantic_score"),
            "Hybrid": (hybrid_frame, "hybrid_score"),
        }

        for model_name, (frame, score_column) in model_frames.items():
            # All models use exactly the same hard filters.
            filtered = apply_constraints(frame, constraints)
            ranked_ids = rank_scores(filtered, score_column)
            print(f"\nQuery: {query} | Model: {model_name}")
            print("Relevant IDs:", sorted(relevant_ids))
            print("Top 3 IDs:", ranked_ids[:3])

            # Evaluate at K=1 and K=3.
            for k in (1, 3):
                results.append({
                    "query": query,
                    "model": model_name,
                    "k": k,
                    "precision": precision_at_k(
                        ranked_ids, relevant_ids, k
                    ),
                    "recall": recall_at_k(
                        ranked_ids, relevant_ids, k
                    ),
                    "ndcg": ndcg_at_k(
                        ranked_ids, relevant_ids, k
                    ),
                    "constraint_violation": 0,
                })

    results_df = pd.DataFrame(results)

    print("\nMetrics for each query:")
    print(results_df.to_string(index=False))

    print("\nAverage metrics by model and K:")
    summary = results_df.groupby(
        ["model", "k"]
    )[["precision", "recall", "ndcg", "constraint_violation"]].mean()

    print(summary.round(3).to_string())

    results_df.to_csv("data/evaluation_results.csv", index=False)
    summary.to_csv("data/evaluation_summary.csv")

    print("\nSaved:")
    print("- data/evaluation_results.csv")
    print("- data/evaluation_summary.csv")


if __name__ == "__main__":
    evaluate()
