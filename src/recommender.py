import pandas as pd
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_movies_dataframe(movies: List[Dict]) -> pd.DataFrame:
    """
    Convert TMDB movie list into a clean DataFrame with the columns we need,
    including poster_path so we can show images in the UI.
    """
    df = pd.DataFrame(movies)

    # Keep poster_path as well
    keep_cols = [
        "id",
        "title",
        "overview",
        "genre_ids",
        "vote_average",
        "vote_count",
        "original_language",
        "poster_path",
    ]
    # Some movies may miss some fields; use intersection
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Fill NaN text fields
    if "overview" in df.columns:
        df["overview"] = df["overview"].fillna("")
    if "original_language" in df.columns:
        df["original_language"] = df["original_language"].fillna("")

    # Genre_ids are numeric; for a simple model we just convert them to strings
    if "genre_ids" in df.columns:
        df["genre_ids"] = df["genre_ids"].apply(
            lambda x: " ".join(map(str, x)) if isinstance(x, list) else ""
        )
    else:
        df["genre_ids"] = ""

    # Build combined text feature
    df["title"] = df["title"].fillna("")
    df["combined_text"] = (
        df["title"] + " " +
        df.get("overview", "") + " " +
        df.get("genre_ids", "") + " " +
        df.get("original_language", "")
    )

    return df

class MovieRecommender:
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["combined_text"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)


    def get_all_titles(self) -> List[str]:
        return self.movies_df["title"].tolist()

    def recommend_by_title(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Return top_n similar movies given a title.
        """
        if title not in self.movies_df["title"].values:
            raise ValueError(f"Title '{title}' not found in catalog")

        # Find index of this movie
        idx = self.movies_df.index[self.movies_df["title"] == title][0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip first one (itself)
        sim_scores = sim_scores[1: top_n + 1]
        indices = [i[0] for i in sim_scores]
        result = self.movies_df.iloc[indices].copy()
        result["similarity"] = [i[1] for i in sim_scores]
        return result

