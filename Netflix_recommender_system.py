import pandas as pd
import numpy as np
import random
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from scipy.sparse.linalg import svds

############################################################
# Data loading
############################################################
# Load movie catalogue
df_mov_titles = pd.read_csv(_Path+'/movie_titles.csv', sep=',', header=None, names=['Movie_Id', 'Year', 'Name'],
                            usecols=[0,1,2], encoding="ISO-8859-1")
df_mov_titles.set_index('Movie_Id', inplace = True)

print('Movies catalogue shape: {}'.format(df_mov_titles.shape))
print(df_mov_titles.head(10))

# Load files with ratings in a loop
_files = ['combined_data_1.txt','combined_data_2.txt','combined_data_3.txt','combined_data_4.txt']
df_ratings = pd.DataFrame()

for _f in _files:
    _staging = pd.read_csv(_Path+'/'+_f, header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
    _staging['Rating'] = _staging['Rating'].astype(float)
    print("{} shape: {}".format(_f, _staging.shape))
    
    if len(df_ratings) > 0:
        df_ratings = pd.concat([df_ratings, _staging], ignore_index=True)
    else:
        df_ratings = _staging
    del _staging
    
print('Tot shape: {}'.format(df_ratings.shape))
print(df_ratings.head())

############################################################
# Data wrangling
############################################################
# Prepare data to place movie id on the columns. (Movie ID is currently a row, where Rating is NaN)

## Create a dataframe keeping the records where the rating is NaN
movies_IDs = pd.DataFrame(pd.isnull(df_ratings.Rating))
movies_IDs = movies_IDs[movies_IDs['Rating'] == True]
movies_IDs = movies_IDs.reset_index() # since movies are in order, we can reset the indexes

## Zip creates tuples: e.g. (548, 0), i.e. at row 548 we have the 1st movie, then from row 548 till 694 the 2nd, etc.
movies_IDs_fin = []
mo = 1 # first movie ID
for i,j in zip(movies_IDs['index'][1:],movies_IDs['index'][:-1]):
    temp = np.full((1,i-j-1), mo) # create an array of size i-j-1 with value mo repeated
    movies_IDs_fin = np.append(movies_IDs_fin, temp)
    mo += 1
## Handle last record (which require len(df_ratings))
last_ = np.full((1,len(df_ratings) - movies_IDs.iloc[-1, 0] - 1), mo)
movies_IDs_fin = np.append(movies_IDs_fin, last_)

print('Movie IDs array shape: {}'.format(movies_IDs_fin.shape))

# Remove rows where rating is NaN and place moviedID as a new column
df_ratings = df_ratings[pd.notnull(df_ratings['Rating'])]

df_ratings['Movie_Id'] = movies_IDs_fin.astype(int)
del movies_IDs_fin
df_ratings['Cust_Id'] = df_ratings['Cust_Id'].astype(int)

print('Tot shape: {}'.format(df_ratings.shape))
print('Tot clients: {}'.format(len(df_ratings['Cust_Id'].unique())))
print(df_ratings.head(3))

############################################################
# Export a data sample
############################################################
num_clients = 32013 # obtained dividing the full client len (480'189) by 15
num_movies = 1300 # obtained dividing the full movie len (17'770) by 15
w_ = [.01, .01, .08, .90] # weights to pick records by quartile, i.e. 1% of clients/movies will be picked by bottom .25 quartile
q_ = [.25, .5, .75, 1.]
random.seed(33)

# Mark movies deciles
movie_summary = df_ratings.groupby('Movie_Id').agg(reviews_count=('Rating','count'))
percentiles_ = np.linspace(0,1,11) # i.e. deciles, labelling each record with the decile it falls into
movie_summary['deciles'] = pd.qcut(movie_summary.reviews_count, percentiles_, labels=percentiles_[:-1])
movie_summary['deciles'] = movie_summary['deciles'].astype('float')
movie_summary.reset_index(inplace=True)
print(movie_summary.head(3))

# Mark customers deciles
cust_summary = df_ratings.groupby('Cust_Id').agg(reviews_count=('Rating','count'))
percentiles_ = np.linspace(0,1,11)
cust_summary['deciles'] = pd.qcut(cust_summary.reviews_count, percentiles_, labels=percentiles_[:-1])
cust_summary['deciles'] = cust_summary['deciles'].astype('float')
cust_summary.reset_index(inplace=True)
print(cust_summary.head(3))

# Pick IDs
movies_IDs = []
cust_IDs = []

qprev = 0.
for i in range(len(w_)):
    mo = random.sample(list(movie_summary.loc[(movie_summary['deciles']>qprev)&(movie_summary['deciles']<=q_[i]), "Movie_Id"]),
                      round(num_movies*w_[i]))
    cus = random.sample(list(cust_summary.loc[(cust_summary['deciles']>qprev)&(cust_summary['deciles']<=q_[i]), "Cust_Id"]),
                       round(num_clients*w_[i]))
    for m in mo:
        movies_IDs.append(m)
    for c in cus:
        cust_IDs.append(c)
    qprev = q_[i] #update previous quantile for next iteration

print("Selected {} movies".format(len(movies_IDs)))
print("Selected {} customers".format(len(cust_IDs)))

# Filter main df
print('Original shape: {}'.format(df_ratings.shape))
df_ratings_XS = df_ratings[df_ratings['Movie_Id'].isin(movies_IDs)]
df_ratings_XS = df_ratings_XS[df_ratings_XS['Cust_Id'].isin(cust_IDs)]
print('After filtering shape: {}'.format(df_ratings_XS.shape))

# Export filtered df
df_ratings_XS.to_csv(_Path+'/df_ratings_XS.txt', header=True, index=True, sep='|', mode='w')
