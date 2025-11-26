import requests
from typing import List, Dict
from .config import TMDB_API_KEY, TMDB_BASE_URL

def _get(path: str, params: Dict = None) -> Dict:
    if params is None:
        params = {}
    params["api_key"] = TMDB_API_KEY
    url = f"{TMDB_BASE_URL}{path}"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

def get_popular_movies(pages: int = 3) -> List[Dict]:
    """
    Fetch popular movies from TMDB.
    pages: how many pages of results to fetch (20 movies per page).
    """
    movies = []
    for page in range(1, pages + 1):
        data = _get("/movie/popular", {"page": page})
        movies.extend(data.get("results", []))
    return movies

def search_movies(query: str, limit: int = 10) -> List[Dict]:
    if not query:
        return []
    data = _get("/search/movie", {"query": query})
    return data.get("results", [])[:limit]

def get_movie_details(movie_id: int) -> Dict:
    return _get(f"/movie/{movie_id}", {"append_to_response": "keywords"})

