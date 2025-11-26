import streamlit as st
import pandas as pd

from src.tmdb_client import get_popular_movies, search_movies, get_movie_details
from src.recommender import build_movies_dataframe, MovieRecommender
from src.config import TMDB_IMAGE_BASE_URL

@st.cache_data(show_spinner=True)
def load_catalog(pages: int = 3) -> pd.DataFrame:
    """
    Fetch popular movies from TMDB and build the DataFrame once.
    Cached so we don't hit the API every time.
    """
    movies = get_popular_movies(pages=pages)
    df = build_movies_dataframe(movies)
    return df

@st.cache_resource(show_spinner=True)
def load_recommender(df: pd.DataFrame) -> MovieRecommender:
    return MovieRecommender(df)

def show_movie_card(movie_row: pd.Series):
    col1, col2 = st.columns([1, 3])
    with col1:
        if pd.notna(movie_row.get("poster_path", None)):
            st.image(TMDB_IMAGE_BASE_URL + movie_row["poster_path"], use_container_width=True)
        else:
            st.write("No image")
    with col2:
        st.subheader(movie_row["title"])
        st.write(f"Rating: {movie_row.get('vote_average', 'N/A')} ({movie_row.get('vote_count', 0)} votes)")
        st.write(f"Language: {movie_row.get('original_language', 'N/A')}")
        st.write(movie_row.get("overview", "")[:400] + "...")

def main():
    st.title("🎬 Live Movie Recommendation System")
    st.write("Recommendations using TMDB online data + TF-IDF & cosine similarity (no local dataset).")

    # Load base catalog and recommender
    with st.spinner("Loading movie catalog and building ML model..."):
        catalog_df = load_catalog(pages=5)   # ~100 movies (5 pages * 20)
        # Fetch details (including poster, overview) for display
        # Optional: augment catalog with poster paths
        # if "poster_path" not in catalog_df.columns:
        #     catalog_df["poster_path"] = None

        recommender = load_recommender(catalog_df)

    option = st.radio("How do you want to select a movie?", ["Choose from popular list", "Search by name"])

    selected_title = None

    if option == "Choose from popular list":
        title_list = recommender.get_all_titles()
        selected_title = st.selectbox("Select a movie:", title_list)
    else:
        query = st.text_input("Search movie by name:")
        if query:
            results = search_movies(query)
            if results:
                titles = [f"{m['title']} ({m.get('release_date', 'N/A')}) - id:{m['id']}" for m in results]
                choice = st.selectbox("Select from search results:", titles)
                if choice:
                    # Extract title piece before " (year)"
                    selected_title = choice.split(" (")[0]
            else:
                st.warning("No results found.")

    if selected_title and st.button("Recommend Similar Movies"):
        try:
            st.subheader(f"Movies similar to: {selected_title}")
            recs = recommender.recommend_by_title(selected_title, top_n=5)

            for _, row in recs.iterrows():
                show_movie_card(row)
                st.markdown("---")
        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
