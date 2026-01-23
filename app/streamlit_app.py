import streamlit as st
import pandas as pd
import numpy as np
import ast

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.recommender.explanations import explain_similarity

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "sample_titles.csv"

df = pd.read_csv(DATA_PATH)

st.set_page_config(
    page_title="Media Recommendation Demo",
    layout="wide"
)

st.title("🎬 Content Recommendation Demo")
st.write(
    "A hybrid recommender combining **story similarity**, "
    "**genre overlap**, and **popularity**."
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/sample_titles.csv")
    df["genres"] = df["genres"].apply(ast.literal_eval)
    return df

df = load_data()

@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df["overview"].fillna("").tolist(),
        show_progress_bar=False
    )
    df["embedding"] = list(embeddings)
    return df

df = load_model_and_embeddings(df)

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
    return df["genres"].apply(
        lambda g: len(target_genres.intersection(set(g)))
    )

def build_user_context(df, liked_title):
    idx = df.index[df["title"] == liked_title][0]

    return {
        "history_titles": [liked_title],
        "history_embeddings": np.array([df.loc[idx, "embedding"]])
    }

def hybrid_recommender(df, title, top_n=5,
                       w_sim=0.6, w_genre=0.2, w_pop=0.2):

    sim = semantic_similarity(df, title)
    genre = genre_overlap(df, title)
    pop = normalize(df["popularity"])

    score = (
        w_sim * normalize(pd.Series(sim)) +
        w_genre * normalize(genre) +
        w_pop * pop
    )

    df["hybrid_score"] = score

    idx = df.index[df["title"] == title][0]

    return (
        df.sort_values("hybrid_score", ascending=False)
          .iloc[1 : top_n + 1]
    )

st.sidebar.header("Recommendation Settings")

selected_title = st.sidebar.selectbox(
    "Select a title",
    df["title"].tolist()
)

w_sim = st.sidebar.slider("Story similarity", 0.0, 1.0, 0.6)
w_genre = st.sidebar.slider("Genre overlap", 0.0, 1.0, 0.2)
w_pop = st.sidebar.slider("Popularity", 0.0, 1.0, 0.2)

st.subheader(f"Recommended if you liked **{selected_title}**")

results = hybrid_recommender(
    df,
    selected_title,
    top_n=5,
    w_sim=w_sim,
    w_genre=w_genre,
    w_pop=w_pop
)

user_context = build_user_context(df, selected_title)

for _, row in results.iterrows():
    st.markdown(f"### 🎥 {row['title']}")
    st.write(f"**Genres:** {', '.join(row['genres'])}")
    st.write(f"**Popularity:** {row['popularity']}")
    st.progress(float(row["hybrid_score"]))

    similar_titles = explain_similarity(
        item_embedding=row["embedding"],
        user_history_embeddings=user_context["history_embeddings"],
        user_history_titles=user_context["history_titles"]
    )

    with st.expander("Why this was recommended"):
        st.write(f"**Similar to:** {', '.join(similar_titles)}")

    st.markdown("---")


