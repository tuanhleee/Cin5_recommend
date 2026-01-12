import numpy as np
import pandas as pd

def getnombreunique(data):
    df = pd.read_csv(data)    
    uniquedirector = df['Director'].nunique()
    uniquegenres = (df['main_genre'] + " " + df['side_genre']).nunique()
    uniqueactors = df['Actors'].nunique()
    return uniquedirector,uniquegenres,uniqueactors

if __name__ == "__main__":
    tf_vectors_file = 'data/dataset.csv'
    size = getnombreunique(tf_vectors_file)
    print(f'Term Frequency Vector Size: {size}')
    print(f'Unique Directors: {size[0]}, Unique Genres: {size[1]}')
    print(f'Unique Actors: {size[2]}')
    
