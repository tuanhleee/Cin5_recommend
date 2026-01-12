import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. PRÉPARATION DES DONNÉES
# ============================================================================

def load_and_prepare_data(filepath):
    """
    Charge et prépare les données pour la recommandation.
    
    Args:
        filepath: Chemin vers le fichier CSV
        
    Returns:
        DataFrame avec les données nettoyées
    """
    data = pd.read_csv(filepath)
    
    # Remplir les valeurs manquantes
    data = data.fillna('')
    
    # S'assurer que Rating et Total_Gross sont numériques
    data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce').fillna(0)
    data['Total_Gross'] = pd.to_numeric(data['Total_Gross'], errors='coerce').fillna(0)
    data['Runtime(Mins)'] = pd.to_numeric(data['Runtime(Mins)'], errors='coerce').fillna(0)
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0)
    
    return data


def create_bag_of_words(data, include_plot=False, genre_weight=1):
    """
    Crée le bag of words pour chaque film.
    
    Args:
        data: DataFrame des films
        include_plot: Inclure le résumé du film
        genre_weight: Nombre de répétitions pour renforcer l'importance des genres
        
    Returns:
        Series contenant le bag of words pour chaque film
    """
    bag = (
        data["Movie_Title"].astype(str) + " " +
        data["Director"].astype(str) + " " +
        data["Actors"].astype(str) + " " +
        (data["main_genre"].astype(str) + " ") * genre_weight +
        (data["side_genre"].astype(str) + " ") * max(1, genre_weight - 1)
    )
    
    if include_plot:
        bag = bag + " " + data["plot"].astype(str)
    
    return bag


def normalize_feature(values):
    """
    Normalise un vecteur de valeurs entre 0 et 1.
    
    Args:
        values: Array numpy de valeurs
        
    Returns:
        Array numpy normalisé
    """
    v_min = values.min()
    v_max = values.max()
    if v_max - v_min == 0:
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


# ============================================================================
# 2. MODÈLES DE RECOMMANDATION
# ============================================================================

def model_content_pure(query_idx, tfidf_matrix, data, top_k=5):
    """
    Modèle 1: Content-Based Pure (baseline)
    Utilise uniquement la similarité de contenu.
    
    Args:
        query_idx: Index du film requête
        tfidf_matrix: Matrice TF-IDF
        data: DataFrame des films
        top_k: Nombre de recommandations
        
    Returns:
        DataFrame des recommandations avec scores
    """
    query_vec = tfidf_matrix[query_idx]
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    
    # Score = similarité uniquement
    scores = sims
    
    return _get_top_recommendations(query_idx, scores, sims, data, top_k)


def model_hybrid_rating(query_idx, tfidf_matrix, data, top_k=5, alpha=0.2):
    """
    Modèle 2: Hybrid Content + Rating
    Combine similarité de contenu et rating.
    
    Args:
        query_idx: Index du film requête
        tfidf_matrix: Matrice TF-IDF
        data: DataFrame des films
        top_k: Nombre de recommandations
        alpha: Poids du rating (0.2 = 20%)
        
    Returns:
        DataFrame des recommandations avec scores
    """
    query_vec = tfidf_matrix[query_idx]
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    
    # Normaliser les ratings
    ratings = data["Rating"].to_numpy()
    rating_norm = normalize_feature(ratings)
    
    # Score hybride
    scores = (1 - alpha) * sims + alpha * rating_norm
    
    return _get_top_recommendations(query_idx, scores, sims, data, top_k)


def model_hybrid_full(query_idx, tfidf_matrix, data, top_k=5, alpha=0.2, beta=0.1):
    """
    Modèle 3: Hybrid Content + Rating + Box Office
    Combine similarité, rating et popularité (box office).
    
    Args:
        query_idx: Index du film requête
        tfidf_matrix: Matrice TF-IDF
        data: DataFrame des films
        top_k: Nombre de recommandations
        alpha: Poids du rating (0.2 = 20%)
        beta: Poids du box office (0.1 = 10%)
        
    Returns:
        DataFrame des recommandations avec scores
    """
    query_vec = tfidf_matrix[query_idx]
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    
    # Normaliser ratings et box office
    ratings = data["Rating"].to_numpy()
    rating_norm = normalize_feature(ratings)
    
    box_office = data["Total_Gross"].to_numpy()
    box_office_norm = normalize_feature(box_office)
    
    # Score hybride complet
    scores = (1 - alpha - beta) * sims + alpha * rating_norm + beta * box_office_norm
    
    return _get_top_recommendations(query_idx, scores, sims, data, top_k)


def model_knn(query_idx, data, top_k=5):
    """
    Modèle 4: K-Nearest Neighbors
    Utilise toutes les features numériques et catégorielles encodées.
    
    Args:
        query_idx: Index du film requête
        data: DataFrame des films
        top_k: Nombre de recommandations
        
    Returns:
        DataFrame des recommandations avec scores
    """
    # Préparer les features numériques
    features_numeric = data[['Year', 'Rating', 'Runtime(Mins)', 'Total_Gross']].copy()
    
    # One-hot encoding des genres
    main_genre_dummies = pd.get_dummies(data['main_genre'], prefix='main')
    side_genre_dummies = pd.get_dummies(data['side_genre'], prefix='side')
    
    # Combiner toutes les features
    features = pd.concat([features_numeric, main_genre_dummies, side_genre_dummies], axis=1)
    
    # Normaliser
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # KNN
    knn = NearestNeighbors(n_neighbors=top_k+1, metric='euclidean')
    knn.fit(features_scaled)
    
    # Trouver les voisins
    distances, indices = knn.kneighbors([features_scaled[query_idx]])
    
    # Exclure le film lui-même
    indices = indices[0][1:]
    distances = distances[0][1:]
    
    # Convertir distances en scores de similarité (inverse)
    max_dist = distances.max() if distances.max() > 0 else 1
    sims = 1 - (distances / max_dist)
    
    # Créer le DataFrame de résultats
    recs = data.iloc[indices][["Movie_Title", "Year", "main_genre", "side_genre", "Rating", "Total_Gross"]].copy()
    recs["similarity"] = sims
    recs["score"] = sims
    
    return recs


def _get_top_recommendations(query_idx, scores, sims, data, top_k):
    """
    Fonction utilitaire pour extraire les top K recommandations.
    
    Args:
        query_idx: Index du film requête
        scores: Scores finaux
        sims: Scores de similarité
        data: DataFrame des films
        top_k: Nombre de recommandations
        
    Returns:
        DataFrame des recommandations
    """
    # Trier par score décroissant
    sorted_idx = np.argsort(scores)[::-1]
    
    # Filtrer pour exclure le film requête
    filtered_idx = [i for i in sorted_idx if i != query_idx][:top_k]
    
    # Créer le DataFrame de résultats
    recs = data.iloc[filtered_idx][["Movie_Title", "Year", "main_genre", "side_genre", "Rating", "Total_Gross"]].copy()
    recs["similarity"] = sims[filtered_idx]
    recs["score"] = scores[filtered_idx]
    
    return recs


# ============================================================================
# 3. ÉVALUATION DES MODÈLES
# ============================================================================

def evaluate_recommendations(recs):
    """
    Évalue la qualité des recommandations selon plusieurs métriques.
    
    Args:
        recs: DataFrame des recommandations
        
    Returns:
        Dict contenant les métriques d'évaluation
    """
    metrics = {
        'avg_rating': recs['Rating'].mean(),
        'avg_similarity': recs['similarity'].mean(),
        'avg_score': recs['score'].mean(),
        'genre_diversity': len(recs['main_genre'].unique()),
        'year_std': recs['Year'].std(),
        'avg_box_office': recs['Total_Gross'].mean()
    }
    
    return metrics


# ============================================================================
# 3.5 VISUALISATIONS
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(aggregated_df):
    """
    Crée des visualisations comparatives pour tous les modèles.
    
    Args:
        aggregated_df: DataFrame avec les résultats agrégés par modèle
    """
    # Configuration du style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Créer une figure avec plusieurs subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comparaison des Modèles de Recommandation', fontsize=20, fontweight='bold')
    
    # 1. Rating moyen
    ax1 = axes[0, 0]
    aggregated_df['avg_rating'].plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Rating Moyen des Recommandations', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Rating (0-10)')
    ax1.set_xlabel('Modèle')
    ax1.axhline(y=aggregated_df['avg_rating'].mean(), color='red', linestyle='--', 
                label=f'Moyenne: {aggregated_df["avg_rating"].mean():.2f}')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Similarité moyenne
    ax2 = axes[0, 1]
    aggregated_df['avg_similarity'].plot(kind='bar', ax=ax2, color='lightgreen', edgecolor='black')
    ax2.set_title('Similarité Moyenne avec le Film Requête', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Similarité (0-1)')
    ax2.set_xlabel('Modèle')
    ax2.axhline(y=aggregated_df['avg_similarity'].mean(), color='red', linestyle='--',
                label=f'Moyenne: {aggregated_df["avg_similarity"].mean():.3f}')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Score hybride moyen
    ax3 = axes[0, 2]
    aggregated_df['avg_score'].plot(kind='bar', ax=ax3, color='salmon', edgecolor='black')
    ax3.set_title('Score Hybride Moyen', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_xlabel('Modèle')
    ax3.axhline(y=aggregated_df['avg_score'].mean(), color='red', linestyle='--',
                label=f'Moyenne: {aggregated_df["avg_score"].mean():.3f}')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Diversité des genres
    ax4 = axes[1, 0]
    aggregated_df['genre_diversity'].plot(kind='bar', ax=ax4, color='plum', edgecolor='black')
    ax4.set_title('Diversité des Genres', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Nombre de genres uniques')
    ax4.set_xlabel('Modèle')
    ax4.axhline(y=aggregated_df['genre_diversity'].mean(), color='red', linestyle='--',
                label=f'Moyenne: {aggregated_df["genre_diversity"].mean():.2f}')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Diversité temporelle (écart-type des années)
    ax5 = axes[1, 1]
    aggregated_df['year_std'].plot(kind='bar', ax=ax5, color='gold', edgecolor='black')
    ax5.set_title('Diversité Temporelle (Écart-type des Années)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Écart-type')
    ax5.set_xlabel('Modèle')
    ax5.axhline(y=aggregated_df['year_std'].mean(), color='red', linestyle='--',
                label=f'Moyenne: {aggregated_df["year_std"].mean():.2f}')
    ax5.legend()
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Box Office moyen
    ax6 = axes[1, 2]
    aggregated_df['avg_box_office'].plot(kind='bar', ax=ax6, color='lightcoral', edgecolor='black')
    ax6.set_title('Box Office Moyen des Recommandations', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Box Office (millions $)')
    ax6.set_xlabel('Modèle')
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    print("📊 Graphique sauvegardé: model_comparison_overview.png")
    plt.show()


def plot_radar_chart(aggregated_df):
    """
    Crée un radar chart pour comparer les modèles sur toutes les métriques.
    
    Args:
        aggregated_df: DataFrame avec les résultats agrégés
    """
    from math import pi
    
    # Normaliser les données entre 0 et 1 pour le radar chart
    df_norm = aggregated_df.copy()
    for col in df_norm.columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val - min_val > 0:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.5
    
    # Préparer les données
    categories = list(df_norm.columns)
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Créer le plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Couleurs pour chaque modèle
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    # Tracer chaque modèle
    for idx, (model_name, row) in enumerate(df_norm.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    # Configurer le graphique
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.set_title('Comparaison Multi-Critères des Modèles\n(Valeurs normalisées)', 
                 size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    print("📊 Graphique sauvegardé: model_comparison_radar.png")
    plt.show()


def plot_heatmap_comparison(results_df):
    """
    Crée une heatmap pour visualiser les performances de chaque modèle sur chaque film test.
    
    Args:
        results_df: DataFrame avec tous les résultats détaillés
    """
    # Créer une matrice pivot pour la heatmap
    pivot_rating = results_df.pivot_table(
        index='model', 
        columns='test_film', 
        values='avg_rating'
    )
    
    pivot_similarity = results_df.pivot_table(
        index='model', 
        columns='test_film', 
        values='avg_similarity'
    )
    
    # Créer les subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Heatmap 1: Rating moyen
    sns.heatmap(pivot_rating, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=axes[0], cbar_kws={'label': 'Rating Moyen'}, linewidths=0.5)
    axes[0].set_title('Rating Moyen par Modèle et Film Test', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Film Test', fontsize=12)
    axes[0].set_ylabel('Modèle', fontsize=12)
    
    # Heatmap 2: Similarité moyenne
    sns.heatmap(pivot_similarity, annot=True, fmt='.3f', cmap='YlGnBu', 
                ax=axes[1], cbar_kws={'label': 'Similarité Moyenne'}, linewidths=0.5)
    axes[1].set_title('Similarité Moyenne par Modèle et Film Test', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Film Test', fontsize=12)
    axes[1].set_ylabel('Modèle', fontsize=12)
    
    plt.tight_layout()
    print("📊 Graphique sauvegardé: model_comparison_heatmap.png")
    plt.show()


def plot_recommendations_example(data, query_idx, recommendations_dict):
    """
    Visualise les recommandations de tous les modèles pour un film donné.
    
    Args:
        data: DataFrame des films
        query_idx: Index du film requête
        recommendations_dict: Dict {model_name: recommendations_df}
    """
    n_models = len(recommendations_dict)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models))
    
    if n_models == 1:
        axes = [axes]
    
    # Film requête
    query_film = data.iloc[query_idx]
    fig.suptitle(f'Recommandations pour: {query_film["Movie_Title"]} ({query_film["Year"]})\n'
                 f'Genre: {query_film["main_genre"]} | Rating: {query_film["Rating"]}',
                 fontsize=16, fontweight='bold')
    
    for idx, (model_name, recs) in enumerate(recommendations_dict.items()):
        ax = axes[idx]
        
        # Préparer les données pour le graphique
        x_labels = [f"{row['Movie_Title'][:20]}..." if len(row['Movie_Title']) > 20 
                    else row['Movie_Title'] for _, row in recs.iterrows()]
        ratings = recs['Rating'].values
        scores = recs['score'].values
        
        # Double axe Y
        x_pos = np.arange(len(x_labels))
        ax2 = ax.twinx()
        
        # Barres pour le rating
        bars1 = ax.bar(x_pos - 0.2, ratings, 0.4, label='Rating', color='skyblue', edgecolor='black')
        # Barres pour le score
        bars2 = ax2.bar(x_pos + 0.2, scores, 0.4, label='Score', color='coral', edgecolor='black')
        
        # Configuration
        ax.set_xlabel('Films Recommandés', fontsize=11)
        ax.set_ylabel('Rating', fontsize=11, color='skyblue')
        ax2.set_ylabel('Score du Modèle', fontsize=11, color='coral')
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.tick_params(axis='y', labelcolor='skyblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        
        # Légendes
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    print("📊 Graphique sauvegardé: recommendations_example.png")
    plt.show()


def compare_models(data, test_indices, models_config):
    """
    Compare tous les modèles sur plusieurs films de test.
    
    Args:
        data: DataFrame des films
        test_indices: Liste des indices de films à tester
        models_config: Dict de configuration des modèles
        
    Returns:
        DataFrame comparatif des résultats
    """
    all_results = []
    
    for test_idx in test_indices:
        for model_name, config in models_config.items():
            # Obtenir les recommandations
            recs = config['function'](**config['params'])
            
            # Évaluer
            metrics = evaluate_recommendations(recs)
            metrics['model'] = model_name
            metrics['test_film'] = data.iloc[test_idx]['Movie_Title']
            metrics['test_idx'] = test_idx
            
            all_results.append(metrics)
    
    return pd.DataFrame(all_results)


def aggregate_results(results_df):
    """
    Agrège les résultats de tous les tests pour comparer les modèles.
    
    Args:
        results_df: DataFrame des résultats individuels
        
    Returns:
        DataFrame avec les moyennes par modèle
    """
    agg = results_df.groupby('model').agg({
        'avg_rating': 'mean',
        'avg_similarity': 'mean',
        'avg_score': 'mean',
        'genre_diversity': 'mean',
        'year_std': 'mean',
        'avg_box_office': 'mean'
    }).round(3)
    
    return agg


# ============================================================================
# 4. FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale pour exécuter la comparaison complète.
    """
    print("🎬 Démarrage de la comparaison des modèles de recommandation...\n")
    
    print("📂 Chargement des données...")
    data = load_and_prepare_data("C:/Users/remis/Documents/M1/cine5/python/data/DATASETULTIME.csv")
    
    # Choisir des films de test variés
    test_indices = [4, 10, 50, 100, 500]
    
    print(f"✅ {len(data)} films chargés")
    print(f"🎯 {len(test_indices)} films de test sélectionnés\n")
    
    # Préparer les bag of words pour différentes configurations
    print("📝 Préparation des bag of words...")
    bow_simple = create_bag_of_words(data, include_plot=False, genre_weight=1)
    bow_with_plot = create_bag_of_words(data, include_plot=True, genre_weight=1)
    bow_weighted = create_bag_of_words(data, include_plot=False, genre_weight=3)
    
    # Vectorisation TF-IDF
    print("🔢 Vectorisation TF-IDF...")
    vectorizer_simple = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_simple = vectorizer_simple.fit_transform(bow_simple)
    
    vectorizer_plot = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_plot = vectorizer_plot.fit_transform(bow_with_plot)
    
    vectorizer_weighted = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_weighted = vectorizer_weighted.fit_transform(bow_weighted)
    
    print("✅ Vectorisation terminée\n")
    
    # Stocker les recommandations pour visualisation
    all_recommendations = {}
    all_results = []
    
    print("🔄 Exécution des modèles sur les films de test...")
    
    for test_idx in test_indices:
        print(f"  → Test sur: {data.iloc[test_idx]['Movie_Title']}")
        
        # Configuration des modèles pour ce test
        models_config = {
            "1_Content_Pure": {
                'function': model_content_pure,
                'params': {
                    'query_idx': test_idx,
                    'tfidf_matrix': tfidf_simple,
                    'data': data,
                    'top_k': 5
                }
            },
            "2_Hybrid_Rating": {
                'function': model_hybrid_rating,
                'params': {
                    'query_idx': test_idx,
                    'tfidf_matrix': tfidf_simple,
                    'data': data,
                    'top_k': 5,
                    'alpha': 0.2
                }
            },
            "3_Hybrid_Full": {
                'function': model_hybrid_full,
                'params': {
                    'query_idx': test_idx,
                    'tfidf_matrix': tfidf_simple,
                    'data': data,
                    'top_k': 5,
                    'alpha': 0.2,
                    'beta': 0.1
                }
            },
            "4_Content_Plot": {
                'function': model_content_pure,
                'params': {
                    'query_idx': test_idx,
                    'tfidf_matrix': tfidf_plot,
                    'data': data,
                    'top_k': 5
                }
            },
            "5_Weighted_Genres": {
                'function': model_hybrid_rating,
                'params': {
                    'query_idx': test_idx,
                    'tfidf_matrix': tfidf_weighted,
                    'data': data,
                    'top_k': 5,
                    'alpha': 0.2
                }
            },
            "6_KNN": {
                'function': model_knn,
                'params': {
                    'query_idx': test_idx,
                    'data': data,
                    'top_k': 5
                }
            }
        }
        
        # Stocker les recommandations pour le premier film (pour visualisation détaillée)
        if test_idx == test_indices[0]:
            all_recommendations[test_idx] = {}
            for model_name, config in models_config.items():
                all_recommendations[test_idx][model_name] = config['function'](**config['params'])
        
        # Comparer les modèles
        results = compare_models(data, [test_idx], models_config)
        all_results.append(results)
    
    print("✅ Exécution terminée\n")
    
    # Combiner tous les résultats
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Agréger les résultats
    print("📊 Agrégation des résultats...")
    aggregated = aggregate_results(results_df)
    
    # Afficher le tableau récapitulatif
    print("\n" + "="*80)
    print("📈 RÉSULTATS AGRÉGÉS (Moyennes sur tous les tests)")
    print("="*80 + "\n")
    print(aggregated.to_string())
    print("\n")
    
    # Générer les visualisations
    print("="*80)
    print("📊 GÉNÉRATION DES VISUALISATIONS")
    print("="*80 + "\n")
    
    print("1️⃣  Création du graphique de comparaison global...")
    plot_model_comparison(aggregated)
    
    print("\n2️⃣  Création du radar chart multi-critères...")
    plot_radar_chart(aggregated)
    
    print("\n3️⃣  Création des heatmaps de performance...")
    plot_heatmap_comparison(results_df)
    
    print("\n4️⃣  Création de l'exemple de recommandations...")
    plot_recommendations_example(data, test_indices[0], all_recommendations[test_indices[0]])
    
    print("\n" + "="*80)
    print("✅ COMPARAISON TERMINÉE !")
    print("="*80)
    print("\n📁 Fichiers générés:")
    print("   • model_comparison_detailed.csv - Résultats détaillés")
    print("   • model_comparison_aggregated.csv - Résultats agrégés")
    print("   • model_comparison_overview.png - Vue d'ensemble des métriques")
    print("   • model_comparison_radar.png - Comparaison multi-critères")
    print("   • model_comparison_heatmap.png - Performance par film test")
    print("   • recommendations_example.png - Exemple de recommandations\n")
    
    return results_df, aggregated


# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    results, aggregated = main()