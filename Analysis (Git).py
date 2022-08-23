import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests
import json
from itertools import product as prod

pd.options.display.width = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 100

with open(r'C:\Users\Te.TE\Documents\Serpent\Data\CSV\Spotify\YourLibrary.json') as f:
    cc = json.load(f)
data = pd.DataFrame(cc['tracks'])
data['uri'] = data['uri'].str.split('spotify:track:').str[1]

# Obtaining audio features from spotify API

# Obtaining spotify API access token
CLIENT_ID = 'xxxxxxxxxxxxxxxx'
CLIENT_SECRET = 'xxxxxxxxxxxxxxxx'

AUTH_URL = 'https://accounts.spotify.com/api/token'
response = requests.post(AUTH_URL,{'grant_type': 'client_credentials','client_id': CLIENT_ID,
                                   'client_secret': CLIENT_SECRET})
response_data = response.json()
token = response_data['access_token']
headers = {'Authorization': 'Bearer {token}'.format(token = token)}

# Obtaining audio features of all songs in Library usin Spotify API
audio_features = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness',
                  'liveness','valence','tempo','duration_ms','popularity']

features_dict = {k: [] for k in audio_features}
for track_uri in data['uri']:
    track_id = f'https://api.spotify.com/v1/audio-features/{track_uri}'
    track = f'https://api.spotify.com/v1/tracks/{track_uri}'

    gg = requests.get(track_id,headers=headers)
    gg2 = requests.get(track,headers=headers)
    j_son = gg.json()
    for key in j_son:
        try:
            features_dict[key].append(j_son[key])
        except KeyError:
            continue
    features_dict['popularity'].append(gg2.json()['popularity'])
    time.sleep(1)

# creating a dataframe using the audio features obtained from the API
features = pd.DataFrame(features_dict)
features['tracks'] = data['tracks'].tolist()
features.to_csv('Library features.csv')

analysis = pd.read_csv('Library features.csv')

# Adding the artist and track name columns from the libray dataframe to the analysis dataframe
analysis.insert(0,'track',data['track'].tolist())
analysis.insert(1,'artist',data['artist'].tolist())

# converting the track_length column from miliseconds to minutes
analysis['track_length'] = round(analysis['duration_ms'] / 60000,2)
analysis.drop(['duration_ms','mode'],axis = 1,inplace = True)

# Am I a hipster?
print(round(analysis['popularity'].mean(),2))  # 49.58

# What is the general mood of songs in the library?
print(np.mean(analysis[['valence','energy','danceability']].mean()))  # 0.6
# slighty high energy and balanced

# See feature correlation
sns.heatmap(analysis.corr(),annot = True,cmap = 'Blues')
plt.subplots_adjust(left = 0.217,bottom = 0.19,right = 0.836,top = 0.917)
plt.xticks(rotation = 55)
plt.savefig('Audio Feature Correlation.png')
# plt.show()

# what is the most negative song in the dataset
print(analysis[analysis['valence'] == analysis['valence'].min()][['track','artist']].to_string(index = False,
                                                                                               header = False))

# What is the most positive song
print(analysis[analysis['valence'] == analysis['valence'].max()][['track','artist']].to_string(index = False,
                                                                                               header = False))

# Histogram distribution of all song features
fig,ax = plt.subplots(nrows = 4,ncols = 3,figsize = (10,5))

# selecting only numeric values (features) from dataset
numeric_data = analysis.select_dtypes(exclude = object)

# creating pairs to be used as subplot indexes
sides = list(prod(range(4),range(3)))
to_plot = dict(zip(sides,numeric_data.columns))

# iterating through each pair and feature name
for key,value in to_plot.items():
    ax[key[0]][key[1]].hist(numeric_data[value],bins = 10)
    ax[key[0]][key[1]].set_title(value.upper(),fontsize = 9,fontstyle = 'italic')

plt.suptitle(f'Distribution of each Feature ({len(analysis)} songs)'.upper())
plt.subplots_adjust(left = 0.1369,bottom = 0.062,right = 0.9,top = 0.917,wspace = 0.274,hspace = 0.502)
plt.savefig('Distribution of Features.png')

# summary analysis for audio features
# analysis.describe().plot(kind='box')

# Boxplot for audio features liveness, valence, energy, speechiness, danceability, acousticness, instrumentalness
x = analysis[analysis.columns[~analysis.columns.isin(['tempo','popularity','key','loudness','track_length'])]]
x.describe().plot(kind = 'box')
plt.xticks(rotation = 45)
plt.ylim(-0.5,1)
plt.tight_layout()
# plt.show()

# Finding Abeerations
print(x.describe())
# plot shows that liveness, speechiness and instrumentalness each have one outlier.
# on investigation of these points, we see that
print(x[x['speechiness'] == x['speechiness'].max()])
print(x[x['liveness'] == x['liveness'].max()])
print(x[x['instrumentalness'] == x['instrumentalness'].max()])

# From our preliminary analysis we found that Ryan Caraveo was the users top artist. We analyze features of his track

caraveo = analysis[analysis['artist'] == 'Ryan Caraveo']

# Are his songs in my library positive or negative?
print(round(caraveo['valence'].mean(),2))  # 0.51

# See feature correlation
sns.heatmap(analysis.corr(),annot = True,cmap = 'Blues')
# plt.show()

# What is the general mood of his songs in the library?
print(round(sum(caraveo[['valence','energy','danceability']].mean()) / 3,2))  # 0.61
