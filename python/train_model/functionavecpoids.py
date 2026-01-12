import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# on peut load les données depuis la bdd
conn = sqlite3.connect("data/films.db")
data = pd.read_sql_query("SELECT * FROM Films", conn)
conn.close()


def recommendation(film_idx, w_director=3.0, w_actors=1.5, w_genres=5.0, w_plot=2.5, w_content=0.88, w_rating=0.08, w_year=0.04):
    """
    y'a juste a changer les poids en argument de la fonction
    y'aura ptet a ouvrir la bdd ici mais ça sera chiant d'ouvrir le dataset en boucle donc peut etre en paramêtre ?
    """
    df = data.set_index('index')

    #total gross sert a rien en vrai
    df = df.drop(columns=["Total_Gross"])

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
    film_position = df.index.get_loc(film_idx)

    print("Film testé:\n", df.loc[film_idx, 'Movie_Title'])
    print("Année", df.loc[film_idx, 'Year'])
    print("Genre", df.loc[film_idx, 'main_genre'], ",", df.loc[film_idx, 'side_genre'], "\n")


    sims_director = cosine_similarity(tfidf_director[film_position], tfidf_director)[0]
    sims_actors = cosine_similarity(tfidf_actors[film_position], tfidf_actors)[0]
    sims_genres = cosine_similarity(tfidf_genres[film_position], tfidf_genres)[0]
    sims_plot = cosine_similarity(tfidf_plot[film_position], tfidf_plot)[0]

    sims = (
        w_director * sims_director +
        w_actors * sims_actors +
        w_genres * sims_genres +
        w_plot * sims_plot
    ) / (w_director + w_actors + w_genres + w_plot)


    ratings = df["Rating"].to_numpy()
    rating_bonus = np.maximum(0, (ratings - 6.0) / 4.0)

    # J'ai essayé d'ajouter un score basé sur la difference d'année de sortie
    query_year = df.loc[film_idx, 'Year']
    years = df["Year"].to_numpy()
    year_diff = np.abs(years - query_year)
    year_proximity = np.maximum(0, 1 - (year_diff / 50))

    #Ici vous pouvez changer les poids pour chaque critère
    scores = (
        w_content * sims + 
        w_rating * rating_bonus + 
        w_year * year_proximity
    )


    sorted_idx = np.argsort(scores)[::-1]
    filtered_idx = [i for i in sorted_idx if i != film_position][:5]


    #return the just the id for the and the score top 5 recommendations
    # add the score to the the return dataframe 
    recofinals = df.iloc[filtered_idx].copy()
    recofinals["score"] = scores[filtered_idx]
    return recofinals

if __name__ == "__main__":
    recommendations = recommendation(film_idx=1)
    print("TOP 5")
    for idx, (i, row) in enumerate(recommendations.iterrows(), 1):
        print(f"{idx}. {row['Movie_Title']} ({row['Year']}) - Genre: {row['main_genre']}, {row['side_genre']} - Rating: {row['Rating']:.1f} - Score: {row['score']:.4f}")
