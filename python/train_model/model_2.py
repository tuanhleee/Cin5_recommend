import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz


conn = sqlite3.connect("C:/Users/remis/Documents/M1/cine5/python/data/movies.db")
data = pd.read_sql_query("SELECT * FROM movies", conn)
conn.close()
print(data.columns)


def load_data_from_db(db_path):
    """Charge les données des films depuis la base de données SQLite."""
    conn = sqlite3.connect(db_path)
    data = pd.read_sql_query("SELECT * FROM movies", conn)
    conn.close()
    return data

def create_text_features(df):
    """Crée les textes combinés pour chaque feature."""
    return {
        'director': df["Director"],
        'actors': df["Actors"],
        'genres': (df["main_genre"]),
        "side_genre":(df["side_genre"]),
        'plot': df["plot"]
        # "title":df["title"]
    }

def create_tfidf_matrices(text_features):
    """Crée les matrices TF-IDF pour chaque feature."""
    vectorizers = {
        'director': TfidfVectorizer( stop_words='english'),
        'actors': TfidfVectorizer( stop_words='english'),
        'genres': TfidfVectorizer( stop_words='english'),
        'plot': TfidfVectorizer( stop_words='english'),
        'side_genre': TfidfVectorizer( stop_words='english')
    }
    
    tfidf_matrices = {}
    for feature_name, vectorizer in vectorizers.items():
        tfidf_matrices[feature_name] = vectorizer.fit_transform(text_features[feature_name])
    
    return tfidf_matrices,vectorizer


def compute_similarities(tfidf_matrices, film_position):
    """Calcule les similarités cosinus pour chaque feature."""
    similarities = {}
    for feature_name, tfidf_matrix in tfidf_matrices.items():
        similarities[feature_name] = cosine_similarity(
            tfidf_matrix[film_position], 
            tfidf_matrix
        )[0]
    return similarities


def compute_combined_similarity(similarities, weights):
    """Calcule la similarité combinée."""
    w_director, w_actors, w_genres, w_side_genres,w_plot = weights
    total_weight = sum(weights)
    
    combined = (
        w_director * similarities['director'] +
        w_actors * similarities['actors'] +
        w_genres * similarities['genres'] +
        w_plot * similarities['plot']+
        w_side_genres + similarities['side_genre']
    ) / total_weight
    
    return combined


def compute_rating_bonus(ratings):
    """Calcule le bonus basé sur le rating."""
    return np.maximum(0, (ratings - 6.0) / 4.0)


def compute_year_proximity(years, query_year):
    """Calcule la proximité temporelle."""
    year_diff = np.abs(years - query_year)
    return np.maximum(0, 1 - (year_diff / 50))


def compute_total_scores(combined_sim, rating_bonus, year_proximity, w_content, w_rating, w_year):
    """Calcule les scores totaux."""
    return w_content * combined_sim + w_rating * rating_bonus + w_year * year_proximity


def get_top_recommendations(scores, film_position, n=5):
    """Retourne les indices des top recommandations."""
    sorted_idx = np.argsort(scores)[::-1]
    filtered_idx = [i for i in sorted_idx if i != film_position][:n]
    return filtered_idx


def create_recommendations_dataframe(df, filtered_idx, scores):
    """Crée le DataFrame des recommandations."""
    recofinals = df.iloc[filtered_idx].copy()
    recofinals["score"] = scores[filtered_idx]
    return recofinals


def create_score_components_dataframe(df, filtered_idx, scores):
    """Crée le DataFrame de décomposition des scores."""
    return pd.DataFrame({
        'film_id': df.iloc[filtered_idx].index,
        'Movie_Title': df.iloc[filtered_idx]['Movie_Title'].values,
        'score_total': scores[filtered_idx],
    })

def recommendation(film_idx,db_path, w_director=3.0, w_actors=1.5, w_genres=5.0,w_side_genres=1.0, 
                   w_plot=2.5, w_content=0.88, w_rating=0.08, w_year=0.04, n=5):
    """
    Fonction de recommandation simplifiée.
    
    Args:
        film_idx: Index du film de référence
        data: DataFrame contenant les données des films
        w_director, w_actors, w_genres, w_plot: Poids pour les features de contenu
        w_content, w_rating, w_year: Poids pour les composantes finales
    
    Returns:
        index_recommandations
    """
    # Préparation
    df = load_data_from_db(db_path)
    text_features = create_text_features(df)
    tfidf_matrices,_ = create_tfidf_matrices(text_features)
    content_weights = (w_director, w_actors, w_genres,w_side_genres, w_plot)
    excluded_indices = film_idx.copy()
    best_idx = []
    rating_bonus = compute_rating_bonus(df["rating"].to_numpy())
    years   = df["year"].to_numpy()
    # Calcul des similarités
    for  index in film_idx :
        
        similarities = compute_similarities(tfidf_matrices, index)
        
        combined_sim = compute_combined_similarity(similarities, content_weights)
        
        # Bonus
        rating_bonus = compute_rating_bonus(df["rating"].to_numpy())
        year_proximity = compute_year_proximity(years, df.loc[index, 'year'])
        
        # Score total
        score= compute_total_scores(combined_sim, rating_bonus, year_proximity, w_content, w_rating, w_year)
        sorted_idx = np.argsort(score)[::-1]

        for i in sorted_idx:
            # print(i)
            if df.index[i] in excluded_indices:
                continue
            else :
                excluded_indices.append(i)
                best_idx.append(i)
                break
    return df.iloc[best_idx]