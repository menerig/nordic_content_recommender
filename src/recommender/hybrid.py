import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

def semantic_similarity(df, title):
    idx = df.index[df["title"] == title][0]
    query_embedding = df.loc[idx, "embedding"].reshape(1, -1)
    all_embeddings = np.vstack(df["embedding"].values)
    return cosine_similarity(query_embedding, all_embeddings)[0]

def genre_overlap(df, title):
    idx = df.index[df["title"] == title][0]
    target_genres = set(df.loc[idx, "genres"])
    return df["genres"].apply(lambda g: len(target_genres.intersection(set(g))))

def hybrid_recommender(df, title, top_n=5, w_sim=0.6, w_genre=0.2, w_pop=0.2):
    sim = semantic_similarity(df, title)
    genre = genre_overlap(df, title)
    pop = normalize(df["popularity"])
    score = (w_sim * normalize(pd.Series(sim)) +
             w_genre * normalize(genre) +
             w_pop * pop)
    df["hybrid_score"] = score
    return df.sort_values("hybrid_score", ascending=False).iloc[1 : top_n + 1]
