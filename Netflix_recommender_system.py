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
