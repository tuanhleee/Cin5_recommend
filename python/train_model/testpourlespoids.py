import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# on peut load les données depuis la bdd
conn = sqlite3.connect(r"C:/Users/remis/Documents/M1/cine5/python/data/films.db")

data = pd.read_sql_query("SELECT * FROM Films", conn)
conn.close()


data = data.set_index('index')

#total gross sert a rien en vrai
data = data.drop(columns=["Total_Gross"])

vectorizer_director = TfidfVectorizer(max_features=1000, stop_words='english')
vectorizer_actors = TfidfVectorizer(max_features=2000, stop_words='english')
vectorizer_genres = TfidfVectorizer(max_features=500, stop_words='english')
vectorizer_plot = TfidfVectorizer(max_features=3000, stop_words='english')


directors_text = data["Director"].astype(str)
actors_text = data["Actors"].astype(str)
genres_text = (data["main_genre"].astype(str) + " ") * 3 + data["side_genre"].astype(str)
plot_text = data["plot"].astype(str)

tfidf_director = vectorizer_director.fit_transform(directors_text)
tfidf_actors = vectorizer_actors.fit_transform(actors_text)
tfidf_genres = vectorizer_genres.fit_transform(genres_text)
tfidf_plot = vectorizer_plot.fit_transform(plot_text)


#ICI VOUS METTEZ L'ID DU FILM QUE VOUS VOULEZ (REGARDER LE CSV CTRL F LE FILM QUE VOUS VOULEZ ET PRENEZ l'ID)
film_idx = 1

film_position = data.index.get_loc(film_idx)

print("Film testé:\n", data.loc[film_idx, 'Movie_Title'])
print("Année", data.loc[film_idx, 'Year'])
print("Genre", data.loc[film_idx, 'main_genre'], ",", data.loc[film_idx, 'side_genre'], "\n")


sims_director = cosine_similarity(tfidf_director[film_position], tfidf_director)[0]
sims_actors = cosine_similarity(tfidf_actors[film_position], tfidf_actors)[0]
sims_genres = cosine_similarity(tfidf_genres[film_position], tfidf_genres)[0]
sims_plot = cosine_similarity(tfidf_plot[film_position], tfidf_plot)[0]


w_director = 3.0
w_actors = 1.5
w_genres = 5.0
w_plot = 2.5

sims = (
    w_director * sims_director +
    w_actors * sims_actors +
    w_genres * sims_genres +
    w_plot * sims_plot
) / (w_director + w_actors + w_genres + w_plot)


ratings = data["Rating"].to_numpy()
rating_bonus = np.maximum(0, (ratings - 6.0) / 4.0)

# J'ai essayé d'ajouter un score basé sur la difference d'année de sortie
query_year = data.loc[film_idx, 'Year']
years = data["Year"].to_numpy()
year_diff = np.abs(years - query_year)
year_proximity = np.maximum(0, 1 - (year_diff / 50))

#Ici vous pouvez changer les poids pour chaque critère
w_content = 0.88
w_rating = 0.08
w_year = 0.04

scores = (
    w_content * sims + 
    w_rating * rating_bonus + 
    w_year * year_proximity
)


sorted_idx = np.argsort(scores)[::-1]
filtered_idx = [i for i in sorted_idx if i != film_position][:5]



print("TOP 5")

recommendations = data.iloc[filtered_idx][["Movie_Title", "Year", "main_genre", "side_genre", "Rating"]].copy()
recommendations["score"] = scores[filtered_idx]

for idx, (i, row) in enumerate(recommendations.iterrows(), 1):
    print(f"{idx}. {row['Movie_Title']} ({row['Year']}) - {row['main_genre']}, {row['side_genre']} - Rating: {row['Rating']:.1f} - Taux de similarité : {row['score']:.4f}")

print("\n" + "="*80)
