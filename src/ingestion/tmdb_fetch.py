import os
import time
import requests
import pandas as pd
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------

API_KEY = os.getenv("TMDB_API_KEY")
if not API_KEY:
    raise RuntimeError("TMDB_API_KEY environment variable not set")

BASE_URL = "https://api.themoviedb.org/3"
DEFAULT_LANGUAGE = "en-US"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------

def _get(endpoint, params=None, sleep=0.25):
    """
    Internal GET helper with rate-limit protection
    """
    params = params or {}
    params["api_key"] = API_KEY

    response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
    response.raise_for_status()

    time.sleep(sleep)  # be nice to the API
    return response.json()


# ----------------------------
# Fetch functions
# ----------------------------

def fetch_popular_movies(pages=5):
    """
    Fetch popular movies from TMDB
    """
    movies = []

    for page in range(1, pages + 1):
        data = _get(
            "movie/popular",
            params={
                "page": page,
                "language": DEFAULT_LANGUAGE
            }
        )
        movies.extend(data["results"])

    return pd.DataFrame(movies)


def fetch_genre_mapping():
    """
    Fetch genre id -> name mapping
    """
    data = _get("genre/movie/list", params={"language": DEFAULT_LANGUAGE})
    return {g["id"]: g["name"] for g in data["genres"]}


# ----------------------------
# Normalization
# ----------------------------

def normalize_movies(df, genre_map):
    """
    Normalize TMDB response into recommender-friendly schema
    """
    df = df.copy()

    df["genres"] = df["genre_ids"].apply(
        lambda ids: [genre_map.get(i, "Unknown") for i in ids]
    )

    normalized = pd.DataFrame({
        "tmdb_id": df["id"],
        "title": df["title"],
        "overview": df["overview"],
        "genres": df["genres"],
        "popularity": df["popularity"],
        "vote_average": df["vote_average"],
        "vote_count": df["vote_count"],
        "language": df["original_language"],
        "release_date": df["release_date"]
    })

    return normalized


# ----------------------------
# Public pipeline
# ----------------------------

def build_tmdb_snapshot(
    pages=5,
    output_file="tmdb_titles.csv"
):
    """
    End-to-end pipeline:
    - fetch popular movies
    - normalize schema
    - save to data/
    """
    print("Fetching TMDB data...")
    raw = fetch_popular_movies(pages)

    print("Fetching genre mapping...")
    genre_map = fetch_genre_mapping()

    print("Normalizing data...")
    normalized = normalize_movies(raw, genre_map)

    output_path = DATA_DIR / output_file
    normalized.to_csv(output_path, index=False)

    print(f"Saved {len(normalized)} titles to {output_path}")
    return normalized


# ----------------------------
# CLI entrypoint
# ----------------------------

if __name__ == "__main__":
    build_tmdb_snapshot(pages=5)
