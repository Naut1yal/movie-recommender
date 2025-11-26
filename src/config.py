import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"

if TMDB_API_KEY is None:
    raise ValueError("TMDB_API_KEY not found.")
