import pandas as pd
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def main(fav_movie = 'iron man'):
    df_movies = pd.read_csv('ml-latest-small/movies.csv')
    df_ratings = pd.read_csv('ml-latest-small/ratings.csv')

    df_ratings = df_ratings.loc[:, df_ratings.columns != 'timestamp']
    df_movies = df_movies.loc[:, df_movies.columns != 'genres']

    df_movie_features = df_ratings.pivot(
        index = 'movieId',
        columns = 'userId',
        values = 'rating'
    ).fillna(0)

    mat_movie_features = csr_matrix(df_movie_features.values)
    
    movie_to_idx = {
        movie: i for i, movie in
        enumerate(list(df_movies.set_index('movieId').loc[df_movie_features.index].title))
    }

    df_movie_features_sparse = csr_matrix(df_movie_features.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model_knn.fit(df_movie_features_sparse)

    def fuzzy_matching(mapper, fav_movie, verbose=True):
        print('inside of fuzzy_matching')
        match_tuple = []

        for title, idx in mapper.items():
            ratio = fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 60:
                match_tuple.append((title,idx,ratio))
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found.')
            return
        if verbose:
            print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
        return match_tuple[0][1]
    
    def make_recommendation(fav_movie, n_recommandations = 10):

        print('inside of make_recommendation')
        listem = []

        print('You have input movie:', fav_movie)
        idx = fuzzy_matching(movie_to_idx, fav_movie, verbose=True)
        #idx = 10
        print('Recommendation system start to make inference')
        print('......\n')

        distances, indices = model_knn.kneighbors(mat_movie_features[idx], n_neighbors=n_recommandations+1)

        raw_recommends = \
            sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        
        reverse_mapper = {v: k for k, v in movie_to_idx.items()}

        print('Recommendations for {}:'.format(fav_movie))
        """for i, (idx,dist) in enumerate(raw_recommends):
            listem.append(['{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], round(dist,2))])
            print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], round(dist,2)))"""

        return enumerate(raw_recommends), reverse_mapper
    print('inside of main')
    raw_recommends, reverse_mapper = make_recommendation(fav_movie)
    return raw_recommends, reverse_mapper
