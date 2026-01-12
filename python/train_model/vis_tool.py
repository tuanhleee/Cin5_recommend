import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# on peut load les données depuis la bdd
conn = sqlite3.connect("python/data/movies.db")
data = pd.read_sql_query("SELECT * FROM Films", conn)
conn.close()


def recommendation(film_idx, w_director=3.0, w_actors=1.5, w_genres=5.0, w_plot=2.5, w_content=0.88, w_rating=0.08, w_year=0.04):
    """
    Fonction de recommandation qui retourne les recommandations avec la décomposition détaillée des scores.
    
    Returns:
        tuple: (recommendations_df, score_components_df)
            - recommendations_df: DataFrame avec les films recommandés et leur score total
            - score_components_df: DataFrame avec la décomposition détaillée des scores
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

    # Calculer la similarité de contenu combinée
    total_weight = w_director + w_actors + w_genres + w_plot
    sims = (
        w_director * sims_director +
        w_actors * sims_actors +
        w_genres * sims_genres +
        w_plot * sims_plot
    ) / total_weight

    # NOUVEAU: Calculer les contributions individuelles de chaque feature au score de contenu
    score_director = w_content * (w_director / total_weight) * sims_director
    score_actors = w_content * (w_actors / total_weight) * sims_actors
    score_genres = w_content * (w_genres / total_weight) * sims_genres
    score_plot = w_content * (w_plot / total_weight) * sims_plot

    ratings = df["Rating"].to_numpy()
    rating_bonus = np.maximum(0, (ratings - 6.0) / 4.0)

    # J'ai essayé d'ajouter un score basé sur la difference d'année de sortie
    query_year = df.loc[film_idx, 'Year']
    years = df["Year"].to_numpy()
    year_diff = np.abs(years - query_year)
    year_proximity = np.maximum(0, 1 - (year_diff / 50))

    # Calculer les composantes individuelles AVANT de les combiner
    score_content = w_content * sims
    score_rating = w_rating * rating_bonus
    score_year = w_year * year_proximity

    # Score total
    scores = score_content + score_rating + score_year


    sorted_idx = np.argsort(scores)[::-1]
    filtered_idx = [i for i in sorted_idx if i != film_position][:5]


    # Créer le DataFrame des recommandations
    recofinals = df.iloc[filtered_idx].copy()
    recofinals["score"] = scores[filtered_idx]
    
    # Créer le DataFrame de décomposition des scores AVEC les sous-composantes
    score_components = pd.DataFrame({
        'film_id': df.iloc[filtered_idx].index,
        'Movie_Title': df.iloc[filtered_idx]['Movie_Title'].values,
        'score_total': scores[filtered_idx],
        'score_content': score_content[filtered_idx],
        'score_rating': score_rating[filtered_idx],
        'score_year': score_year[filtered_idx],
        # Sous-composantes du contenu
        'score_director': score_director[filtered_idx],
        'score_actors': score_actors[filtered_idx],
        'score_genres': score_genres[filtered_idx],
        'score_plot': score_plot[filtered_idx],
        # Similarités brutes pour référence
        'sim_director': sims_director[filtered_idx],
        'sim_actors': sims_actors[filtered_idx],
        'sim_genres': sims_genres[filtered_idx],
        'sim_plot': sims_plot[filtered_idx],
        'sim_content_combined': sims[filtered_idx]
    })
    
    return recofinals, score_components


def create_dashboard(film_idx, recommendations, score_components, df, params):
    """
    Crée un dashboard pour visualiser les recommandations de films.
    
    Args:
        film_idx: ID du film de référence
        recommendations: DataFrame retourné par la fonction recommendation() (premier élément du tuple)
        score_components: DataFrame avec la décomposition des scores (deuxième élément du tuple)
        df: DataFrame complet des films
        params: dictionnaire contenant les paramètres utilisés
                {'w_director', 'w_actors', 'w_genres', 'w_plot', 'w_content', 'w_rating', 'w_year'}
    """
    # Récupérer les informations du film de référence
    film_ref = df.loc[film_idx]
    
    # Recalculer les composantes des scores pour la décomposition
    film_position = df.index.get_loc(film_idx)
    
    # Créer la figure avec subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ==================== TITRE ET INFOS DU FILM ====================
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    title_text = f"SYSTÈME DE RECOMMANDATION - Film de référence: {film_ref['Movie_Title']}"
    ax_title.text(0.5, 0.8, title_text, ha='center', va='top', 
                  fontsize=18, fontweight='bold')
    
    film_info = f"Année: {film_ref['Year']} | Genre: {film_ref['main_genre']}, {film_ref['side_genre']} | Note: {film_ref['Rating']:.1f}/10"
    ax_title.text(0.5, 0.5, film_info, ha='center', va='top', fontsize=12)
    
    # Paramètres utilisés
    params_text = (f"Poids: Director={params['w_director']:.2f}, Actors={params['w_actors']:.2f}, "
                   f"Genres={params['w_genres']:.2f}, Plot={params['w_plot']:.2f}\n"
                   f"Alpha: Content={params['w_content']:.2f}, Rating={params['w_rating']:.2f}, "
                   f"Year={params['w_year']:.2f}")
    ax_title.text(0.5, 0.2, params_text, ha='center', va='top', 
                  fontsize=10, style='italic', color='gray')
    
    # ==================== TABLEAU DES RECOMMANDATIONS ====================
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    # Préparer les données du tableau
    table_data = []
    for idx, (i, row) in enumerate(recommendations.iterrows(), 1):
        table_data.append([
            f"{idx}",
            row['Movie_Title'][:30] + '...' if len(row['Movie_Title']) > 30 else row['Movie_Title'],
            f"{row['Year']}",
            f"{row['main_genre']}, {row['side_genre']}"[:25],
            f"{row['Rating']:.1f}",
            f"{row['score']:.4f}"
        ])
    
    table = ax_table.table(cellText=table_data,
                          colLabels=['#', 'Titre', 'Année', 'Genres', 'Note', 'Score'],
                          cellLoc='left',
                          loc='center',
                          colWidths=[0.05, 0.35, 0.1, 0.25, 0.1, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Styliser l'en-tête
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alterner les couleurs des lignes
    for i in range(1, 6):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    # ==================== GRAPHIQUE SCORES TOTAUX ====================
    ax_scores = fig.add_subplot(gs[2, 0])
    
    film_titles = [row['Movie_Title'][:20] + '...' if len(row['Movie_Title']) > 20 
                   else row['Movie_Title'] for _, row in recommendations.iterrows()]
    scores = recommendations['score'].values
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(scores)))
    bars = ax_scores.barh(film_titles, scores, color=colors)
    
    ax_scores.set_xlabel('Score Total', fontsize=11, fontweight='bold')
    ax_scores.set_title('Scores Totaux des Recommandations', fontsize=12, fontweight='bold')
    ax_scores.invert_yaxis()
    
    # Ajouter les valeurs sur les barres
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax_scores.text(score + 0.01, i, f'{score:.4f}', 
                      va='center', fontsize=9)
    
    ax_scores.grid(axis='x', alpha=0.3, linestyle='--')
    
    # ==================== GRAPHIQUE DÉCOMPOSITION DES SCORES ====================
    ax_decomp = fig.add_subplot(gs[2, 1])

    # Récupérer les sous-composantes du contenu
    score_director = score_components['score_director'].values
    score_actors = score_components['score_actors'].values
    score_genres = score_components['score_genres'].values
    score_plot = score_components['score_plot'].values
    score_rating = score_components['score_rating'].values
    score_year = score_components['score_year'].values

    # Créer les barres empilées avec les 4 features de contenu + rating + year
    bar_width = 0.6
    indices = np.arange(len(film_titles))

    # Les 4 composantes du contenu (en dégradé de bleu)
    p1 = ax_decomp.barh(indices, score_director, bar_width, 
                        label='Director', color='#1565C0')
    p2 = ax_decomp.barh(indices, score_actors, bar_width, 
                        left=score_director, label='Actors', color='#1976D2')
    p3 = ax_decomp.barh(indices, score_genres, bar_width, 
                        left=score_director + score_actors, label='Genres', color='#42A5F5')
    p4 = ax_decomp.barh(indices, score_plot, bar_width, 
                        left=score_director + score_actors + score_genres, label='Plot', color='#90CAF9')

    # Rating et Year (couleurs différentes)
    content_total = score_director + score_actors + score_genres + score_plot
    p5 = ax_decomp.barh(indices, score_rating, bar_width, 
                        left=content_total, label='Rating', color='#4CAF50')
    p6 = ax_decomp.barh(indices, score_year, bar_width, 
                        left=content_total + score_rating, label='Year', color='#FF9800')

    # Ajouter les valeurs dans les zones colorées (seulement si assez large)
    for i in indices:
        cumul = 0
        
        # Director
        if score_director[i] > 0.02:  # Seuil minimum pour afficher
            ax_decomp.text(cumul + score_director[i]/2, i, f'{score_director[i]:.3f}', 
                        ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        cumul += score_director[i]
        
        # Actors
        if score_actors[i] > 0.02:
            ax_decomp.text(cumul + score_actors[i]/2, i, f'{score_actors[i]:.3f}', 
                        ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        cumul += score_actors[i]
        
        # Genres
        if score_genres[i] > 0.02:
            ax_decomp.text(cumul + score_genres[i]/2, i, f'{score_genres[i]:.3f}', 
                        ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        cumul += score_genres[i]
        
        # Plot
        if score_plot[i] > 0.02:
            ax_decomp.text(cumul + score_plot[i]/2, i, f'{score_plot[i]:.3f}', 
                        ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        cumul += score_plot[i]
        
        # Rating
        if score_rating[i] > 0.02:
            ax_decomp.text(cumul + score_rating[i]/2, i, f'{score_rating[i]:.3f}', 
                        ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        cumul += score_rating[i]
        
        # Year
        if score_year[i] > 0.02:
            ax_decomp.text(cumul + score_year[i]/2, i, f'{score_year[i]:.3f}', 
                        ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax_decomp.set_yticks(indices)
    ax_decomp.set_yticklabels(film_titles)
    ax_decomp.set_xlabel('Contribution au Score', fontsize=11, fontweight='bold')
    ax_decomp.set_title('Décomposition Détaillée des Scores\n(Les 4 nuances de bleu = composantes du "Contenu")', 
                        fontsize=12, fontweight='bold')
    ax_decomp.legend(loc='lower right', fontsize=8, ncol=2)
    ax_decomp.invert_yaxis()
    ax_decomp.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    recommendations, score_components = recommendation(film_idx=1)
    
    print("TOP 5")
    for idx, (i, row) in enumerate(recommendations.iterrows(), 1):
        print(f"{idx}. {row['Movie_Title']} ({row['Year']}) - Genre: {row['main_genre']}, {row['side_genre']} - Rating: {row['Rating']:.1f} - Score: {row['score']:.4f}")
    
    print("\n" + "="*80)
    print("DÉCOMPOSITION DES SCORES:")
    print("="*80)
    for idx, row in score_components.iterrows():
        print(f"\n{row['Movie_Title']}:")
        print(f"  Score Total: {row['score_total']:.4f}")
        print(f"  └─ Contenu:  {row['score_content']:.4f} ({row['score_content']/row['score_total']*100:.1f}%)")
        print(f"  └─ Note:     {row['score_rating']:.4f} ({row['score_rating']/row['score_total']*100:.1f}%)")
        print(f"  └─ Année:    {row['score_year']:.4f} ({row['score_year']/row['score_total']*100:.1f}%)")

        # Paramètres
    params = {
        'w_director': 3.0,
        'w_actors': 1.5,
        'w_genres': 5.0,
        'w_plot': 2.5,
        'w_content': 0.88,
        'w_rating': 0.08,
        'w_year': 0.04
    }
    
    film_idx = 1
    df = data.set_index('index')
    
    # Obtenir les recommandations ET la décomposition des scores
    recommendations, score_components = recommendation(
        film_idx=film_idx,
        w_director=params['w_director'],
        w_actors=params['w_actors'],
        w_genres=params['w_genres'],
        w_plot=params['w_plot'],
        w_content=params['w_content'],
        w_rating=params['w_rating'],
        w_year=params['w_year']
    )
    
    # Créer le dashboard (il faut maintenant passer aussi score_components)
    fig = create_dashboard(film_idx, recommendations, score_components, df, params)
    plt.show()
