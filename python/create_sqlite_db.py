import sqlite3
import pandas as pd
import os
from pathlib import Path
import numpy as np
# BASE_DIR = "/home/tuanh/M1/projet/code/code/python"
BASE_DIR = Path(__file__).resolve().parent
csv_path = os.path.join(BASE_DIR, "./data/movies.csv")
db_path = os.path.join(BASE_DIR, "./data/movies.db")


data = pd.read_csv(csv_path)
data = data.fillna(" ")
print(np.count_nonzero(data.isna()))
required_cols = [
    "Movie_Title",
    "Director",
    "Actors",
    "Year",
    "main_genre",
    "side_genre",
    "Rating",
    "plot",
    "image_filename",
]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    raise ValueError(f"missing value: {missing}")


data = data.reset_index().rename(columns={"index": "id"})

conn = sqlite3.connect(db_path)
cur = conn.cursor()


cur.execute("""
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY,
    title TEXT,
    year INTEGER,
    Director TEXT,
    Actors TEXT,
    main_genre TEXT,
    side_genre TEXT,
    rating REAL,
    plot TEXT,
    image_filename TEXT
)
""")


rows = []
for _, row in data.iterrows():
    movie_id = int(row["id"])
    title = str(row["Movie_Title"]) if not pd.isna(row["Movie_Title"]) else ""
    year = int(row["Year"]) if not pd.isna(row["Year"]) else None
    director = str(row["Director"]) if not pd.isna(row["Director"]) else ""
    actors = str(row["Actors"]) if not pd.isna(row["Actors"]) else ""
    main_genre = str(row["main_genre"]) if not pd.isna(row["main_genre"]) else ""
    side_genre = str(row["side_genre"]) if not pd.isna(row["side_genre"]) else ""
    rating = float(row["Rating"]) if not pd.isna(row["Rating"]) else None
    plot = str(row["plot"]) if not pd.isna(row["plot"]) else ""
    image_filename = str(row["image_filename"]) if not pd.isna(row["image_filename"]) else ""

    rows.append((
        movie_id,
        title,
        year,
        director,
        actors,
        main_genre,
        side_genre,
        rating,
        plot,
        image_filename,
    ))

print(f"Nombre de films à insérer: {len(rows)}")

cur.executemany("""
INSERT OR REPLACE INTO movies
(id, title, year, Director, Actors, main_genre, side_genre, rating, plot, image_filename)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", rows)

conn.commit()
print("Insertion terminée.")


cur.execute("SELECT COUNT(*) FROM movies")
count = cur.fetchone()[0]
print("Nombre de lignes dans movies:", count)

cur.execute("""
SELECT id, title, year, Director, Actors, main_genre, side_genre, rating, plot, image_filename
FROM movies
LIMIT 5
""")
for r in cur.fetchall():
    print(r)

conn.close()
print("SQLite fermé.")
