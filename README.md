# Nordic Content Recommendation Engine (Hybrid, Explainable)

A hybrid recommendation system designed to surface **personalized content for Nordic and Scandinavian audiences**, combining collaborative filtering, content-based embeddings, and explainable ranking.

This project demonstrates how data science can support **content discovery, audience engagement, and retention** for streaming and media platforms operating in the Nordic region.

---

## 🎯 Motivation

Most recommender systems struggle with:
- culturally specific niches (e.g. Nordic noir, Swedish comedy)
- multilingual audiences (Swedish / English / Nordic languages)
- cold-start titles with limited interaction data

This project explores a **hybrid approach** that blends:
- user behavior (implicit feedback)
- semantic understanding of content (text embeddings)
- regional and cultural metadata

The result is a system that can recommend *why* something fits a viewer — not just *what* to watch.

---

## 🧠 System Overview

**Architecture**
## 🧠 Key Components
- Implicit-feedback collaborative filtering (ALS)
- Content embeddings from descriptions and subtitles
- Approximate nearest-neighbour search (FAISS)
- Learning-to-rank model (tree-based, e.g. gradient boosting)
- Explanation layer (feature-based and similarity-based)

---

## 📊 Data Sources

This project uses **publicly available data** and **simulated user interactions** to enable end-to-end experimentation.

### Metadata
- **TMDB** – titles, genres, keywords, cast, crew, language
- **IMDb** – ratings, popularity, vote counts

### Text
- Title overviews
- Subtitles (OpenSubtitles or similar public datasets)

### User Interaction
- Synthetic user-event logs generated from:
  - popularity curves
  - genre affinity
  - regional availability

> ⚠️ No proprietary or private streaming data is used.

---

## 🔍 Recommendation Approach

### 1. Candidate Generation

**Collaborative filtering**
- Implicit ALS trained on watch-time signals

**Content-based retrieval**
- Semantic similarity using sentence-transformer embeddings
- FAISS index for fast nearest-neighbour search

Candidate sets from both approaches are merged before ranking.

---

### 2. Ranking

A learning-to-rank model scores each candidate using:
- user–item latent similarity
- embedding cosine similarity
- genre overlap
- recency and popularity signals
- region and language match

---

### 3. Explainability

Each recommendation includes:
- nearest-neighbour references
- dominant metadata features
- optional SHAP-based explanations for ranking decisions

---

## 📈 Evaluation

Offline evaluation:
- Recall@K
- NDCG@K
- Mean Reciprocal Rank (MRR)

Simulated online metrics:
- click probability
- session length
- recommendation diversity

---

## 🖥 Demo Application

A lightweight **Streamlit app** demonstrates:
- user profile selection
- personalized recommendations
- explanation bubbles (“Why this?”)
- filtering by language, genre, and region

Run locally:
```bash
streamlit run app/streamlit_app.py
