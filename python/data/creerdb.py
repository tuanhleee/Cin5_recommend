import sqlite3
import pandas as pd

df = pd.read_csv('python/data/dataset.csv')
conn = sqlite3.connect('python/data/movies.db')
df.to_sql('Films', conn, if_exists='replace', index=False)
conn.close()