import json
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score,GridSearchCV,validation_curve
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import my_functions
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

pd.options.display.width = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 100


with open(r'C:\Users\Te.TE\Documents\Serpent\Data\CSV\Spotify\YourLibrary.json',encoding = 'UTF-8') as f:
    cc = json.load(f)
data = pd.DataFrame(cc['tracks'])

# Using the obtained features in the last section
my_library = pd.read_csv('Library features.csv')
my_library.insert(0,'track',data['track'].tolist())
my_library.insert(1,'artist',data['artist'].tolist())
my_library['track_length'] = round(my_library['duration_ms'] / 60000,2)
my_library.drop(['duration_ms','mode'],axis = 1,inplace = True)
# my_library.drop([15,131,78,155],axis = 0,inplace = True)

# using KMeans to classify tracks in Library
# we will classify the songs based on valence, loudness, energy, danceability

features = my_library[['valence','speechiness','energy','danceability']]

# Transforming the data
my_library_scaled = MinMaxScaler().fit_transform(features)

# PLotting the elbow graph to aid in deciding optimal number of classes using a predefined function
# my_functions.elbow(scaled, marker = 'x',save = True,show = False,name = 'Features_Elbow.png')

km = KMeans(n_clusters = 3,random_state = 1)
predicted_classes = km.fit_predict(my_library_scaled)

my_library['class'] = predicted_classes

labels = [f'Class {i}' for i in my_library['class'].unique()]

plt.pie(my_library['class'].value_counts(),labels = labels,explode = [0.03,0.03,0.05],
        colors = sns.color_palette('dark'),autopct = '%.0f%%')
plt.title('Class Distribution',fontweight = 'bold',fontsize = 12)
plt.savefig('Class Distribution.png')

# mean features for each class
class_feature_mean = my_library.pivot_table(index = 'class',values = features.columns,aggfunc = 'mean')
class_feature_mean.T.plot(kind = 'bar',edgecolor = 'black')
plt.legend(loc = 1)
plt.xticks(ticks = np.arange(4),labels = class_feature_mean.columns,fontsize = 10,rotation = 0)
plt.xlabel('Features',fontweight = 'bold',fontsize = 12)
plt.ylabel('Mean',fontweight = 'bold',fontsize = 12)
plt.savefig('Mean Features for Each Class')

# Boxpots for each class
fig,ax = plt.subplots(nrows = 1,ncols = 3,figsize = (10,5),sharey = 'all')

col = 0
for group,stuff in my_library.groupby('class'):
    ax[col].boxplot(stuff[['valence','speechiness','energy','danceability']].describe())
    ax[col].set_xticks(ticks = np.arange(1,5),labels = ['valence','speechiness','energy','danceability'],
                       fontsize = 10,rotation = 33)
    ax[col].set_ylim([0,1])
    col += 1
plt.savefig('Boxplots')

# Creating new dataframe from playlists obtained from kaggle**
files = glob(r'C:\Users\Te.TE\Documents\Serpent\Data\CSV\Spotify\Kaggle\*.csv')
kaggle = pd.concat((pd.read_csv(file,encoding = 'UTF-8') for file in files),ignore_index = True)
# print(kaggle.shape)  # (12910, 22)

kaggle['track_length'] = round(kaggle['duration_ms'] / 60000,2)

# Dropping duplicate tracks using track id
kaggle.drop_duplicates(subset = 'id',inplace = True)
kaggle.drop(['Genres','id','uri','track_href','analysis_url','time_signature','mode','Playlist','duration_ms'],
            axis = 1,inplace = True)
kaggle.rename(columns = {'Artist Name':'artist','Track Name':'track','Popularity':'popularity'},inplace = True)
# print(kaggle.shape)  # (9064, 14)

# How many songs in my library are in the kaggle set
intersect = kaggle.reset_index().merge(my_library[['artist','track']], how='inner', on=['artist', 'track'])
intersect.set_index('index',inplace = True)  # EGW1

'''First we want to train a model using our library dataframe and use to group the songs in the new dataframe'''
# Also using the same features we used in clustering

kaggle_transform = MinMaxScaler().fit_transform(kaggle[['valence','speechiness','energy','danceability']])

model_features = my_library_scaled
model_target = np.ravel(my_library[['class']])

# Finding best hyperparameters
hyper = GridSearchCV(estimator = KNeighborsClassifier(),param_grid = {'n_neighbors': [3,5,7,9,11],
                                                                      'weights': ['uniform','distance']})
hyper.fit(model_features,model_target)
# print(f'Best score of {round(hyper.best_score_,4)} with parameters {hyper.best_params_}')

# Plotting the effect of hyperparameters
param = np.arange(3,13,2)
train_scores,test_scores = validation_curve(estimator = KNeighborsClassifier(),X = model_features,y = model_target,
                                            param_name = 'n_neighbors',
                                            param_range = param,cv = 5,scoring = "accuracy")

plt.plot(param,np.mean(test_scores,axis = 1),label = "Test score",color = "blue")
plt.annotate('Peak',xy = (7.01,0.95104),
             xytext = (6.5,0.95),
             va = 'center',
             ha = 'right',
             arrowprops = {'arrowstyle': '-|>','lw': 2},
             fontweight = 'bold')
plt.xlabel('n_neighbors')
plt.ylabel('Model Score')
plt.savefig('Effect of n_neighbors on model accuracy.png')

knn = KNeighborsClassifier(n_neighbors = 7)

Q1,Q2,A1,A2 = train_test_split(model_features,model_target,random_state = 1,stratify = my_library['class'],
                               train_size = 0.65)
knn.fit(Q1,A1)
con = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(A2,knn.predict(Q2)),display_labels = knn.classes_)
con.plot(cmap='Blues')
# plt.show()

# predict using kaggle_transform
kaggle_pred = knn.predict(kaggle_transform)
kaggle['class'] = kaggle_pred

# successfully classified songs in the kaggle dataset using my own defined genres (classes)

plt.pie(kaggle['class'].value_counts(),labels = labels,explode = [0.03,0.03,0.05],
        colors = sns.color_palette('dark'),autopct = '%.0f%%')
plt.title('Class Distribution',fontweight = 'bold',fontsize = 12)
plt.savefig('Class Distribution of Kaggle Set.png')

# print(len(kaggle[kaggle['valence'] < 0.3]))

# Item-item filtering collaborative recommender
# Build Reccomender using cosine similarity

# Avoid recommending songs found in both libraries
kaggle.drop(index = intersect.index,inplace = True)  # NW1

# Selecting features to be used for
use = my_library.columns[~my_library.columns.isin(['artist','track','Popularity','track_length','key','mode'])]
x_scaled = pd.DataFrame(MinMaxScaler().fit_transform(my_library[use]))
y_scaled = pd.DataFrame(MinMaxScaler().fit_transform(kaggle[use]))

# 1: Similarity of Songs within Library
lib_cosim = pd.DataFrame(cosine_similarity(x_scaled,dense_output = False),columns = my_library['track'].values,
                         index = my_library['track'].values)
# print(lib_cosim['Daisies'].drop(index = 'Daisies').sort_values(ascending = False).head(10))

# Handling repeated data in index
my_library['song_&_artist'] = my_library['track'] + ': ' + my_library['artist']

final = pd.DataFrame()
kag_cosim = pd.DataFrame(cosine_similarity(x_scaled,y_scaled),columns = kaggle[['track','artist']].values,
                         index = my_library['song_&_artist'].values)

for song_name in kag_cosim.index:
    timp = kag_cosim.loc[song_name].sort_values(ascending = False).head()
    song = [song_name for i in range(5)]
    vv = pd.DataFrame(list(zip(song,timp.index,timp.values)),columns = ['Library_Track','Similar_Songs','Similarity'])
    final = pd.concat([final,vv])

final.set_index(['Library_Track',pd.MultiIndex.from_tuples(final['Similar_Songs'],names = ['Similar_Songs','Artist'])],
                inplace = True)
final.drop(columns = 'Similar_Songs',inplace = True)

# Finding the most recommended artists and songs
artists = final.index.get_level_values('Artist')
sound = final.index.get_level_values('Similar_Songs')
# print(artists.value_counts()[artists.value_counts() > 3])
# print(sound.value_counts()[sound.value_counts() > 3])

# Convert to Excel File
with pd.ExcelWriter('output.xlsx') as writer:
    final.to_excel(writer,engine ='xlsxwriter', encoding = 'UTF-8')

def find_from_my_library(song_name_,artist):
    """
    :param song_name_: name of song in string format
    :param artist: name of artist in string format
    :return: Series object
    Raises:
    KeyError – If any items are not found.
    """

    if song_name_.title() not in my_library['track'].tolist():
        raise KeyError(f'{song_name_.title ()} is not in your library')
    if artist.title() not in my_library['artist'].tolist():
        raise KeyError(f'{artist.title()} is not in your library')

    print(final.loc[f'{song_name_.title()}: {artist.title()}'])
    
