import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Load in the data
df = pd.read_csv('ml-20m/ratings.csv')
print(df.head())

df.movieId = pd.Categorical(df.movieId)
df['new_movie_id'] = df.movieId.cat.codes # new_movie_id is the new movie id after mapping
df.userId = pd.Categorical(df.userId)
df['new_user_id'] = df.userId.cat.codes # new_user_id is the new user id after mapping
