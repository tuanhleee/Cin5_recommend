from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import random
from typing import List, Optional
import pickle
import pandas as pd
from rapidfuzz import process, fuzz
# =========================
# Config & global objects
# =========================
app = FastAPI()
BASE_URL = "https://res.cloudinary.com/ds84b9f8s/image/upload/v1763570455/movies/"
BASE_DIR = Path(__file__).resolve().parent

# Load TF-IDF model (matrices + vectorizers)
with open(BASE_DIR / "train_model" / "recommender_model.pkl", "rb") as f:
    model_data = pickle.load(f)
tfidf_matrices = model_data["tfidf_matrices"]
vectorizers = model_data["vectorizers"]

# Load SQLite DB into a DataFrame
DB_PATH = BASE_DIR / "data" / "movies.db"
conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
cur = conn.cursor()
df = pd.read_sql_query("SELECT * FROM movies", conn)


N_MOVIES = len(df)

IMAGE_DEFAUT = "img_5555.jpg"


class MovieOut(BaseModel):
    id: int
    title: str
    year: int
    Director: str
    Actors: str
    rating: float
    main_genre: str
    side_genre: str
    plot: str
    image_filename: str

    similarity: Optional[float] = None
    score: Optional[float] = None


class RecoRequest(BaseModel):
    movie_ids: List[int]  # indices 0..N_MOVIES-1
    # weights model
    top_k: int = 5 
    w_director: float = 3.0
    w_actors: float = 1.5
    w_genres: float = 5.0
    w_side_genres: float = 1.0
    w_plot: float = 2.5
    w_content: float = 0.88
    w_rating: float = 0.08
    w_year: float = 0.04


class Request_film(BaseModel):
    films: int = 10


class Find_film(BaseModel):
    Movie_Title: Optional[str] = None
    Movie_director: Optional[str] = None
    Movie_Actors: Optional[str] = None
    Movie_genre: Optional[str] = None
    Rating: Optional[float] = None
    Year: Optional[int] = None
    films: int = 10


def check_img(link_img: str) -> str:
    if not link_img:
        return BASE_URL + IMAGE_DEFAUT
    if link_img.startswith("http://") or link_img.startswith("https://"):
        return link_img
    return BASE_URL + link_img


def compute_similarities(tfidf_matrices_dict, film_position: int):
    """Compute cosine similarities for each TF-IDF feature."""
    similarities = {}
    for feature_name, tfidf_matrix in tfidf_matrices_dict.items():
        similarities[feature_name] = cosine_similarity(
            tfidf_matrix[film_position],
            tfidf_matrix
        )[0]
    return similarities


def compute_combined_similarity(similarities, weights):
    """Combine content similarities with given weights."""
    w_director, w_actors, w_genres, w_side_genres, w_plot = weights
    total_weight = float(sum(weights))

    combined = (
        w_director * similarities["director"]
        + w_actors * similarities["actors"]
        + w_genres * similarities["genres"]
        + w_side_genres * similarities["side_genre"]
        + w_plot * similarities["plot"]
    ) / total_weight

    return combined


def compute_rating_bonus(ratings: np.ndarray):
    """Simple monotone bonus based on rating."""
    return np.maximum(0.0, (ratings - 6.0) / 4.0)


def compute_year_proximity(years: np.ndarray, query_year: float):
    """Temporal proximity to the query year."""
    year_diff = np.abs(years - query_year)
    return np.maximum(0.0, 1.0 - (year_diff / 50.0))


def compute_total_scores(
    combined_sim: np.ndarray,
    rating_bonus: np.ndarray,
    year_proximity: np.ndarray,
    w_content: float,
    w_rating: float,
    w_year: float,
):
    """Final score = content + rating + year."""
    return w_content * combined_sim + w_rating * rating_bonus + w_year * year_proximity

def recommend_ids(
    movie_ids: List[int],
    top_k: int = 5,
    content_weights: tuple = (3.0, 1.5, 5.0, 1.0, 2.5),
    weights_model: tuple = (0.88, 0.08, 0.04),
):

    """
    Simple recommender:
      - For each seed movie mid, compute a score vector.
      - Pick the best candidate not already in movie_ids.
      - Return best_idx, similarities and scores aligned with best_idx.
    """

    w_content,w_rating,w_year = weights_model

    excluded = set(movie_ids)
    best_idx: List[int] = []
    best_sims: List[float] = []
    best_scores: List[float] = []

    ratings = df["rating"].to_numpy()
    years = df["year"].to_numpy()

    rating_bonus = compute_rating_bonus(ratings)

    for mid in movie_ids:
        if mid < 0 or mid >= N_MOVIES:
            raise ValueError(f"Invalid movie index in movie_ids: {mid}")

        # 1) Content similarities
        sims = compute_similarities(tfidf_matrices, mid)
        combined_sim = compute_combined_similarity(sims, content_weights)

        # 2) Year proximity
        year_proximity = compute_year_proximity(years, years[mid])

        # 3) Total score
        score = compute_total_scores(
            combined_sim,
            rating_bonus,
            year_proximity,
            w_content,
            w_rating,
            w_year
        )

        sorted_idx = np.argsort(score)[::-1]

        for i in sorted_idx:
            if i not in excluded:
                excluded.add(i)
                best_idx.append(int(i))
                best_sims.append(float(combined_sim[i]))
                best_scores.append(float(score[i]))
                break

        if len(best_idx) >= top_k:
            break

    return best_idx, best_sims, best_scores


def df_to_json(df_subset: pd.DataFrame):
    """Convert a DataFrame subset to JSON-friendly list of dicts with correct image URLs."""
    df_copy = df_subset.copy()
    df_copy["image_filename"] = df_copy["image_filename"].apply(check_img)
    records = df_copy.to_dict(orient="records")
    return records



def _has_text(x) -> bool:
    if x is None:
        return False
    s = str(x).strip()
    if s == "" or s.lower() == "string":
        return False
    return True


def _fuzzy_scores(query, series, min_score=60):
    """
    get simialire_fuzzy form db.
    """
    values = series.tolist()
    matches = process.extract(
        query,
        values,
        scorer=fuzz.partial_ratio,
        limit=len(values)
    )

    sims = np.zeros(len(values), dtype=float)
    any_good = False

    for val, score, rel_idx in matches:
        if score >= min_score:
            sims[rel_idx] = score / 100.0
            any_good = True

    if not any_good:
        return None
    return sims


# =========================
#  FastAPI endpoints
# =========================
@app.get("/")
def root():
    return {"message": "Film recommendation API (TF-IDF + SQLite)"}

@app.post("/find_film", response_model=List[MovieOut])
def find_film(request: Find_film):

    if N_MOVIES == 0:
        raise HTTPException(status_code=500, detail="Base de films vide.")

    ratings = df["rating"].to_numpy(dtype=float)
    years   = df["year"].to_numpy(dtype=int)

    
    K = np.arange(N_MOVIES)

    
    if request.Rating is not None:
        thr = float(request.Rating)
        K = K[ratings[K] >= thr]
        if len(K) == 0:
            raise HTTPException(
                status_code=404,
                detail="Aucun film avec un rating suffisant."
            )


    if request.Year is not None and request.Year != 0:
        year_req = int(request.Year)
        K = K[years[K] == year_req]
        if len(K) == 0:
            raise HTTPException(
                status_code=404,
                detail="Aucun film pour cette année."
            )


    has_text = any([
        _has_text(request.Movie_Title),
        _has_text(request.Movie_director),
        _has_text(request.Movie_Actors),
        _has_text(request.Movie_genre),
    ])

    
    if not has_text:
        K_sorted = K[np.argsort(ratings[K])[::-1]]
        top_idx  = K_sorted[: request.films].tolist()
        best_film = df.iloc[top_idx]
        if best_film.empty:
            raise HTTPException(status_code=404, detail="Aucun film trouvé")
        return df_to_json(best_film)

    # =========================
    # 2) Fuzzy similarity
    # =========================

    
    K_work = K.copy()
    sims_K = np.zeros(len(K_work), dtype=float)
    weight_sum = 0.0

    # Title
    if _has_text(request.Movie_Title):
        sims_title = _fuzzy_scores(
            request.Movie_Title,
            df["title"].iloc[K_work],
            min_score=50
        )

        if sims_title is not None:
            
            mask = sims_title > 0
            if mask.any():
                
                K_work = K_work[mask]
                sims_K = sims_title[mask].copy()
                weight_sum = 1.0
            else:
                # No title match strong enough -> ignore title criterion
                pass

# -----  Director -----
    if _has_text(request.Movie_director):
        sims_dir = _fuzzy_scores(
            request.Movie_director,
            df["Director"].iloc[K_work],
            min_score=60
        )
        if sims_dir is not None:
            if weight_sum == 0:
                sims_K = sims_dir.copy()
            else:
                sims_K += sims_dir
            weight_sum += 1.0

    # ----- Actors -----
    if _has_text(request.Movie_Actors):
        sims_act = _fuzzy_scores(
            request.Movie_Actors,
            df["Actors"].iloc[K_work],
            min_score=60
        )
        if sims_act is not None:
            if weight_sum == 0:
                sims_K = sims_act.copy()
            else:
                sims_K += sims_act
            weight_sum += 1.0

    # ----- 2.4 Genres (combine main + side) -----
    if _has_text(request.Movie_genre):
        genres_series = (
                df["main_genre"].fillna("") + " " +
                df["side_genre"].fillna("")
            ).iloc[K_work]

        sims_gen = _fuzzy_scores(
            request.Movie_genre,
            genres_series,
            min_score=60
        )
        if sims_gen is not None:
            if weight_sum == 0:
                sims_K = sims_gen.copy()
            else:
                sims_K += sims_gen
            weight_sum += 1.0

    # If no similarity could be computed -> fallback to rating only
    if weight_sum == 0:
        K_sorted = K[np.argsort(ratings[K])[::-1]]
        top_idx = K_sorted[: request.films].tolist()
        best_film = df.iloc[top_idx].copy()
        best_film["similarity"] = 0.0
        return df_to_json(best_film)

    # Average similarity over all used text criteria
    sims_K /= weight_sum

    # 3) Optionally blend with rating
    r_min, r_max = ratings.min(), ratings.max()
    if r_max > r_min:
        rating_norm = (ratings - r_min) / (r_max - r_min)
    else:
        rating_norm = np.ones_like(ratings)

    alpha = 0.2  
    score_K = (1 - alpha) * sims_K + alpha * rating_norm[K_work]

    # 4) Sort by final score
    order = np.argsort(score_K)[::-1][: request.films]

    top_idx = K_work[order].tolist()
    top_scores = score_K[order].tolist()

    best_film = df.iloc[top_idx].copy()
    if best_film.empty:
        raise HTTPException(status_code=404, detail="Aucun film trouvé")

    best_film["similarity"] = top_scores
    return df_to_json(best_film)

@app.post("/recommend", response_model=List[MovieOut])
def recommend(request: RecoRequest):
    """
    request.movie_ids = list of seed indices (0..N_MOVIES-1).
    Returns top_k recommended films, excluding the seeds.
    """
    content_weights = (request.w_director,request.w_actors,request.w_genres, request.w_side_genres,request.w_plot)
    weights_model= (request.w_content,request.w_rating,request.w_year)
    try:
        top_idx, sims, scores = recommend_ids(
        movie_ids=request.movie_ids,
        top_k=request.top_k,
        content_weights=content_weights,
        weights_model=weights_model,
)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not top_idx:
        raise HTTPException(status_code=404, detail="Aucune recommandation trouvée")

    best_film = df.iloc[top_idx].copy()
    if best_film.empty:
        raise HTTPException(status_code=404, detail="Aucun film trouvé")

    # sims and scores are aligned with top_idx
    best_film["similarity"] = sims
    best_film["score"] = scores

    return df_to_json(best_film)


@app.post("/get_film", response_model=List[MovieOut])
def get_film(request: Request_film):
    """
    Draw 'films' random films from [0..N_MOVIES-1].
    """
    if request.films <= 0 or request.films > N_MOVIES:
        raise HTTPException(status_code=400, detail="Nombre de films invalide")

    random_ids = random.sample(range(N_MOVIES), request.films)

    best_film = df.iloc[random_ids]
    if best_film.empty:
        raise HTTPException(status_code=404, detail="Aucun film trouvé")

    return df_to_json(best_film)

@app.get("/get_info")
def get_info():
    """
    Get info: min/max year, genres, side_genres
    """
    year = df["year"]
    min_year = min(year)
    max_year = max(year)
    genre = set(df["main_genre"])
    side_genre = set(df["side_genre"])

    return {
        "min_year": min_year ,
        "max_year": max_year,
        "genres":genre,
        "side_genres":side_genre
        }