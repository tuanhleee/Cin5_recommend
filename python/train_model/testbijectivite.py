import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


conn = sqlite3.connect('python/data/movies.db')
data = pd.read_sql_query("SELECT * FROM Films", conn)
conn.close()

df = data.set_index('index')
df = df.drop(columns=["Total_Gross"])

#ça calcul les tf idf ici pour que ça soit fait une seul fois
vectorizer_director = TfidfVectorizer(max_features=1000, stop_words='english')
vectorizer_actors = TfidfVectorizer(max_features=2000, stop_words='english')
vectorizer_genres = TfidfVectorizer(max_features=500, stop_words='english')
vectorizer_plot = TfidfVectorizer(max_features=3000, stop_words='english')

directors_text = df["Director"].astype(str)
actors_text = df["Actors"].astype(str)
genres_text = (df["main_genre"].astype(str) + " ") * 3 + df["side_genre"].astype(str)
plot_text = df["plot"].astype(str)

tfidf_director = vectorizer_director.fit_transform(directors_text)
tfidf_actors = vectorizer_actors.fit_transform(actors_text)
tfidf_genres = vectorizer_genres.fit_transform(genres_text)
tfidf_plot = vectorizer_plot.fit_transform(plot_text)


sim_director = cosine_similarity(tfidf_director)
sim_actors = cosine_similarity(tfidf_actors)
sim_genres = cosine_similarity(tfidf_genres)
sim_plot = cosine_similarity(tfidf_plot)

ratings = df["Rating"].to_numpy()
rating_bonus = np.maximum(0, (ratings - 6.0) / 4.0)
years = df["Year"].to_numpy()
movie_ids = df.index.tolist()


def test_bijectivite(w_director=3.0, w_actors=1.5, w_genres=5.0, w_plot=2.5, w_content=0.88, w_rating=0.08, w_year=0.04):

    total_weight = w_director + w_actors + w_genres + w_plot
    
    sims = (w_director * sim_director + w_actors * sim_actors + 
            w_genres * sim_genres + w_plot * sim_plot) / total_weight
    
    bijcount = 0
    
    for movie_id in movie_ids:
        film_position = df.index.get_loc(movie_id)
        
        query_year = years[film_position]
        year_diff = np.abs(years - query_year)
        year_proximity = np.maximum(0, 1 - (year_diff / 50))
        
        scores = w_content * sims[film_position] + w_rating * rating_bonus + w_year * year_proximity
        
        sorted_idx = np.argsort(scores)[::-1]
        top_recommendation_position = next(idx for idx in sorted_idx if idx != film_position)
        
        query_year_reverse = years[top_recommendation_position]
        year_diff_reverse = np.abs(years - query_year_reverse)
        year_proximity_reverse = np.maximum(0, 1 - (year_diff_reverse / 50))
        
        scores_reverse = w_content * sims[top_recommendation_position] + w_rating * rating_bonus + w_year * year_proximity_reverse
        sorted_idx_reverse = np.argsort(scores_reverse)[::-1]
        top5_positions = [idx for idx in sorted_idx_reverse if idx != top_recommendation_position][:5]
        
        if film_position in top5_positions:
            bijcount += 1
    
    rate = bijcount / len(movie_ids) * 100
    return bijcount, rate


# cfg
test_configs = [
    {"name": "base", "w_director": 3.0, "w_actors": 1.5, "w_genres": 5.0, "w_plot": 2.5, 
     "w_content": 0.88, "w_rating": 0.08, "w_year": 0.04},
    
    {"name": "Genre-Heavy", "w_director": 2.0, "w_actors": 1.0, "w_genres": 8.0, "w_plot": 2.0, 
     "w_content": 0.88, "w_rating": 0.08, "w_year": 0.04},
    
    {"name": "Director-Heavy", "w_director": 6.0, "w_actors": 1.5, "w_genres": 3.0, "w_plot": 2.0, 
     "w_content": 0.88, "w_rating": 0.08, "w_year": 0.04},
    
    {"name": "Balanced", "w_director": 2.5, "w_actors": 2.5, "w_genres": 2.5, "w_plot": 2.5, 
     "w_content": 0.88, "w_rating": 0.08, "w_year": 0.04},
    
    {"name": "Rating-Heavy", "w_director": 3.0, "w_actors": 1.5, "w_genres": 5.0, "w_plot": 2.5, 
     "w_content": 0.70, "w_rating": 0.25, "w_year": 0.05},
    
    {"name": "Plot-Heavy", "w_director": 1.0, "w_actors": 1.0, "w_genres": 3.0, "w_plot": 8.0, 
     "w_content": 0.88, "w_rating": 0.08, "w_year": 0.04},
    
    {"name": "Year-Heavy", "w_director": 3.0, "w_actors": 1.5, "w_genres": 5.0, "w_plot": 2.5, 
     "w_content": 0.70, "w_rating": 0.08, "w_year": 0.22},
    
    {"name": "Actor-Heavy", "w_director": 1.0, "w_actors": 7.0, "w_genres": 3.0, "w_plot": 2.0, 
     "w_content": 0.88, "w_rating": 0.08, "w_year": 0.04},
]



results = []

for config in test_configs:
    name = config["name"]
    config_copy = {k: v for k, v in config.items() if k != "name"}
    bijcount, rate = test_bijectivite(**config_copy)
    results.append((name, bijcount, rate))


results.sort(key=lambda x: x[2], reverse=True)
for name, count, rate in results:
    print(f"{name:20s}: {count:4d} films ({rate:5.2f}%)")