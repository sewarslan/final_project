import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
#for time calculating
from datetime import datetime


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Starting Time =", current_time)

#read data
df_movies = pd.read_csv('../dataset/ml-25m/movies.csv')
df_ratings = pd.read_csv('../dataset/ml-25m/ratings.csv')

#drop timestamp, genres and name columns
df_ratings = df_ratings.loc[:, df_ratings.columns != 'timestamp']
df_ratings.dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float64'}
df_movies = df_movies.loc[:, df_movies.columns != 'genres']
df_movies.dtype={'movieId': 'int32', 'title': 'str'}

#find unique user count and unique movie count
num_users = len(df_ratings.userId.unique())
num_items = len(df_ratings.movieId.unique())
print('There are {} unique users and {} unique movies in this data set'.format(num_users, num_items))

#getting how many ratings given to rating values
df_ratings_cnt_tmp = pd.DataFrame(df_ratings.groupby('rating').size(), columns=['count'])
df_ratings_cnt_tmp

#finding the number of all elements on csr matrix
total_cnt = num_users * num_items
#finding the number of ratings that supposed to be zero in csr matrix
rating_zero_cnt = total_cnt - df_ratings.shape[0]
# append counts of zero rating to df_ratings_cnt
df_ratings_cnt = df_ratings_cnt_tmp.append(
    pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
    verify_integrity=True,
).sort_index()
df_ratings_cnt


# add log count
df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])
df_ratings_cnt

# get rating frequency
df_movies_cnt = pd.DataFrame(df_ratings.groupby('movieId').size(), columns=['count'])
df_movies_cnt.head()

# filter data
#popularity_thres = 200
popular_movies = list(set(df_movies_cnt.query('count >= 200').index))
df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]
print('shape of original ratings data: ', df_ratings.shape)
print('shape of ratings data after dropping unpopular movies: ', df_ratings_drop_movies.shape)

# get number of ratings given by every user
df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])

# filter data
#ratings_thres = 200
#drop inactive users by setting a threshold
active_users = list(set(df_users_cnt.query('count >= 200').index))
df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]
print('shape of original ratings data: ', df_ratings.shape)
print('shape of ratings data after dropping both unpopular movies and inactive users: ', df_ratings_drop_users.shape)

# pivot and create movie-user matrix which its values are ratings
movie_user_mat = df_ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# create mapper from movie title to index
movie_to_idx = {
    movie: i for i, movie in 
    enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title))
}

# transform matrix to scipy sparse matrix
movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

#since our data has high dimensionality we will be using cosine similarity instead of euclidean
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(movie_user_mat_sparse)

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

def make_recommendation(fav_movie, n_recommandations = 10):
    
    print('inside of make_recommendation')
    listem = []

    print('You have input movie:', fav_movie)
    idx = fuzzy_matching(fav_movie)
    #idx = 10
    print('Recommendation system start to make inference')
    print('....../n')

    distances, indices = model_knn.kneighbors(movie_user_mat_sparse[idx], n_neighbors=n_recommandations+1)

    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    
    reverse_mapper = {v: k for k, v in movie_to_idx.items()}

    """print('Recommendations for {}:'.format(fav_movie))
    for i, (idx,dist) in enumerate(raw_recommends):
        listem.append(['{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], round(dist,2))])
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], round(dist,2)))"""

    return enumerate(raw_recommends), reverse_mapper





now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Ending Time =", current_time)