from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import sqlite3


def load_data_from_db(db_path):
    """Charge les données des films depuis la base de données SQLite."""
    conn = sqlite3.connect(db_path)
    data = pd.read_sql_query("SELECT * FROM Films", conn)
    conn.close()
    return data


def prepare_dataframe(data):
    """Prépare le DataFrame en supprimant les colonnes inutiles."""
    df = data.set_index('index')
    df = df.drop(columns=["Total_Gross"])
    return df


def create_text_features(df):
    """Crée les textes combinés pour chaque feature."""
    return {
        'director': df["Director"].astype(str),
        'actors': df["Actors"].astype(str),
        'genres': (df["main_genre"].astype(str) + " ") * 3 + df["side_genre"].astype(str),
        'plot': df["plot"].astype(str)
    }

# travail sur max_features
def create_tfidf_matrices(text_features):
    """Crée les matrices TF-IDF pour chaque feature."""
    vectorizers = {
        'director': TfidfVectorizer(max_features=None, stop_words='english'),
        'actors': TfidfVectorizer(max_features=None, stop_words='english'),
        'genres': TfidfVectorizer(max_features=None, stop_words='english'),
        'plot': TfidfVectorizer(max_features=3000, stop_words='english')
    }
    
    tfidf_matrices = {}
    for feature_name, vectorizer in vectorizers.items():
        tfidf_matrices[feature_name] = vectorizer.fit_transform(text_features[feature_name])
    
    return tfidf_matrices


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
    w_director, w_actors, w_genres, w_plot = weights
    total_weight = sum(weights)
    
    combined = (
        w_director * similarities['director'] +
        w_actors * similarities['actors'] +
        w_genres * similarities['genres'] +
        w_plot * similarities['plot']
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

def recommendation(film_idx, data, w_director=3.0, w_actors=1.5, w_genres=5.0, 
                   w_plot=2.5, w_content=0.88, w_rating=0.08, w_year=0.04, n=5):
    """
    Fonction de recommandation simplifiée.
    
    Args:
        film_idx: Index du film de référence
        data: DataFrame contenant les données des films
        w_director, w_actors, w_genres, w_plot: Poids pour les features de contenu
        w_content, w_rating, w_year: Poids pour les composantes finales
    
    Returns:
        tuple: (recommendations_df, score_components_df)
    """
    # Préparation
    df = prepare_dataframe(data)
    text_features = create_text_features(df)
    tfidf_matrices = create_tfidf_matrices(text_features)
    film_position = df.index.get_loc(film_idx)
    
    # Calcul des similarités
    similarities = compute_similarities(tfidf_matrices, film_position)
    content_weights = (w_director, w_actors, w_genres, w_plot)
    combined_sim = compute_combined_similarity(similarities, content_weights)
    
    # Bonus
    rating_bonus = compute_rating_bonus(df["Rating"].to_numpy())
    year_proximity = compute_year_proximity(df["Year"].to_numpy(), df.loc[film_idx, 'Year'])
    
    # Score total
    scores = compute_total_scores(combined_sim, rating_bonus, year_proximity, w_content, w_rating, w_year)
    
    # Recommandations
    filtered_idx = get_top_recommendations(scores, film_position, n)
    recommendations_df = create_recommendations_dataframe(df, filtered_idx, scores)
    score_components_df = create_score_components_dataframe(df, filtered_idx, scores)
    
    return recommendations_df, score_components_df

def model_v1_one_per_film(films_idx, data):
    """
    Version 1: Retourne une recommandation par film en entrée (exactement 5).
    
    Prend le meilleur film recommandé pour chaque film en entrée.
    Si un film est déjà présent dans la liste, prend le suivant.
    
    Args:
        films_idx: Liste de 5 indices de films
        data: DataFrame contenant les données des films
    
    Returns:
        list: Liste de 5 indices de films recommandés (un par film en entrée)
    """
    if len(films_idx) != 5:
        raise ValueError("La liste doit contenir exactement 5 films")
    
    result_list = []
    
    for film_idx in films_idx:
        recommendations_df, _ = recommendation(film_idx, data)
        
        # Parcourir les recommandations dans l'ordre jusqu'à trouver un film pas encore sélectionné
        for _, row in recommendations_df.iterrows():
            recommended_idx = row.name  # L'index du film recommandé
            
            # Vérifier que le film n'est pas déjà dans la liste de résultats
            if recommended_idx not in result_list and recommended_idx not in films_idx:
                result_list.append(recommended_idx)
                break
    
    return result_list


def model_v2_best_overall(films_idx, data):
    """
    Version 2: Retourne les 5 meilleurs films recommandés globalement.
    
    Regroupe toutes les recommandations, élimine les doublons en gardant 
    le meilleur score, et retourne les 5 films avec les meilleurs scores.
    
    Args:
        films_idx: Liste d'indices de films (typiquement 5)
        data: DataFrame contenant les données des films
    
    Returns:
        list: Liste de 5 indices de films recommandés avec les meilleurs scores
    """
    all_recommendations = {}  # {film_idx: best_score}
    
    for film_idx in films_idx:
        recommendations_df, _ = recommendation(film_idx, data)
        
        # Parcourir toutes les recommandations
        for _, row in recommendations_df.iterrows():
            recommended_idx = row.name
            score = row['score']
            
            # Garder le meilleur score pour chaque film
            if recommended_idx not in all_recommendations:
                all_recommendations[recommended_idx] = score
            else:
                all_recommendations[recommended_idx] = max(all_recommendations[recommended_idx], score)
    
    # Exclure les films qui sont dans la liste d'entrée
    for film_idx in films_idx:
        all_recommendations.pop(film_idx, None)
    
    # Trier par score décroissant et prendre les 5 meilleurs
    sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
    top_5 = [film_idx for film_idx, score in sorted_recommendations[:5]]
    
    return top_5

def display_input_films(films_idx, data):
    """
    Affiche les films en entrée de manière formatée.
    
    Args:
        films_idx: Liste d'indices de films
        data: DataFrame contenant les données des films
    """
    df = prepare_dataframe(data)
    
    print("=" * 80)
    print(f"FILMS EN ENTRÉE ({len(films_idx)} films)")
    print("=" * 80)
    
    for i, film_idx in enumerate(films_idx, 1):
        film = df.loc[film_idx]
        print(f"\n{i}. {film['Movie_Title']}")
        print(f"   Année: {film['Year']}")
        print(f"   Genre: {film['main_genre']}, {film['side_genre']}")
        print(f"   Rating: {film['Rating']}/10")
        print(f"   Réalisateur: {film['Director']}")
    
    print("\n" + "=" * 80 + "\n")


def display_output_films(films_idx, data, scores=None):
    """
    Affiche les films recommandés en sortie de manière formatée.
    
    Args:
        films_idx: Liste d'indices de films recommandés
        data: DataFrame contenant les données des films
        scores: Dict optionnel {film_idx: score} pour afficher les scores
    """
    df = prepare_dataframe(data)
    
    print("=" * 80)
    print(f"FILMS RECOMMANDÉS ({len(films_idx)} films)")
    print("=" * 80)
    
    for i, film_idx in enumerate(films_idx, 1):
        film = df.loc[film_idx]
        print(f"\n{i}. {film['Movie_Title']}")
        print(f"   Année: {film['Year']}")
        print(f"   Genre: {film['main_genre']}, {film['side_genre']}")
        print(f"   Rating: {film['Rating']}/10")
        print(f"   Réalisateur: {film['Director']}")
        
        if scores and film_idx in scores:
            print(f"   Score de recommandation: {scores[film_idx]:.4f}")
    
    print("\n" + "=" * 80 + "\n")

# Charger les données
data = load_data_from_db("python/data/movies.db")
films_idx = [12, 45, 78, 123, 456]

# Afficher les films en entrée
display_input_films(films_idx, data)

# Obtenir les recommandations (version 1)
result1 = model_v1_one_per_film(films_idx, data)

# Afficher les recommandations
display_output_films(result1, data)

# Obtenir les recommandations (version 2)
result2 = model_v2_best_overall(films_idx, data)

# Afficher les recommandations
display_output_films(result2, data)