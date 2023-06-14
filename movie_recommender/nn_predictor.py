import numpy as np
from numpy import genfromtxt
import pickle
from fuzzywuzzy import fuzz

with open('./pickles/movie_to_idx.pickle', 'rb') as f:
    # Dump the dictionary into the file using pickle
    movie_to_idx = pickle.load(f)

with open('./pickles/m_dist.pickle', 'rb') as f:
    # Dump the dictionary into the file using pickle
    m_dist = pickle.load(f)

with open('./pickles/movie_dict.pickle', 'rb') as f:
    # Dump the dictionary into the file using pickle
    movie_dict = pickle.load(f)

item_vecs = genfromtxt('csv/item_vecs.csv', delimiter=',')

def fuzzy_matching(fav_movie, verbose=True):
    print('inside of fuzzy matching')
    """
    return the closest match via fuzzy ratio. If no match found, return None
    
    Parameters
    ----------    
    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie
    
    verbose: bool, print log if True

    Return
    ------
    index of the closest match
    """
    match_tuple = []
    # get match
    for title, idx in movie_to_idx.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]

def make_pred(fav_movie, n_recommandations = 10):
    id_listesi = []
    title_listesi = []
    idx= fuzzy_matching(fav_movie)
    for i in np.argsort(m_dist[idx])[::1][:n_recommandations]:
        movid = int(item_vecs[i,0])
        id_listesi.append(movid)
        title_listesi.append(movie_dict[item_vecs[i,0]]['title'])


    return id_listesi, title_listesi