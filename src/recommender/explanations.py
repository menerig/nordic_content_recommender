import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def explain_similarity(
    item_embedding,
    user_history_embeddings,
    user_history_titles,
    top_k=2
):
    sims = cosine_similarity(
        item_embedding.reshape(1, -1),
        user_history_embeddings
    )[0]

    top_idx = np.argsort(sims)[-top_k:][::-1]

    return [user_history_titles[i] for i in top_idx]


def explain_metadata(item_row, user_profile):
    reasons = []

    if item_row["genre"] in user_profile["top_genres"]:
        reasons.append(f"You often watch {item_row['genre']} titles")

    if item_row["language"] == user_profile["language"]:
        reasons.append(f"Matches your preferred language ({item_row['language']})")

    return reasons


def generate_explanation(item_row, user_context):
    explanation = {}

    explanation["similar_titles"] = explain_similarity(
        item_row["embedding"],
        user_context["history_embeddings"],
        user_context["history_titles"]
    )

    explanation["metadata"] = explain_metadata(
        item_row, user_context
    )

    return explanation
