#!/usr/bin/env python
# coding: utf-8

# In[98]:


# %%writefile test.py

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
# from matplotlib.lines import Line2D
import pickle  
import time
from sklearn.cluster import KMeans

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random
from bs4 import BeautifulSoup
import requests
import re


# In[6]:


# Spotify credentials:

input_file = open("/Users/renev/OneDrive/Desktop/input.txt","r")
string = input_file.read()
secrets_dict={}
for line in string.split('\n'):
    if len(line) > 0:
        secrets_dict[line.split(':')[0]]=line.split(':')[1]
auth_manager = SpotifyClientCredentials(client_id = secrets_dict['client_id'], 
                                        client_secret = secrets_dict['client_secret'])
sp = spotipy.Spotify(auth_manager=auth_manager)


# In[7]:


# Load the model
filename = 'song_model.sav'
kmeans = pickle.load(open(filename, 'rb'))


# In[8]:


#Load the scaler   
scaler = pickle.load(open('scaler.pkl', 'rb'))


# In[9]:


# Load the database of songs
database_songs = pd.read_csv('database_songs.csv')


# In[10]:


database_songs


# In[11]:


database_songs['clusters'].value_counts()


# In[99]:


# User input → based on Spotify song url to avoid mis-typing, etc.

user_input = input("Please enter a Spotify track url: ")


# In[100]:


track_id = user_input[31:53]


# In[101]:


track_id


# Collect the audio features from the Spotify API. 

# In[102]:


#     track_info = sp.track(id)
#     track_features = sp.audio_features(id)

track_info = sp.track(track_id)
track_features = sp.audio_features(track_id)
    
#     Track info
name = track_info['name']
album= track_info['album']['name']
artist= track_info['album']['artists'][0]['name']
#     release_date= track_info['album']['release_date']
#     length= track_info['duration_ms']
#     popularity= track_info['popularity']
    
#     Track features

try: 
    danceability = track_features[0]['danceability']
    energy=track_features[0]['energy']
#     key=track_features[0]['key']
    loudness= track_features[0]['loudness']
#     mode=track_features[0]['mode']
    speechiness=track_features[0]['speechiness']
    acousticness= track_features[0]['acousticness']
    instrumentalness=track_features[0]['instrumentalness']
    liveness=track_features[0]['liveness']
    valence= track_features[0]['valence']
    tempo=track_features[0]['tempo']
    id= track_features[0]['id']
    duration_ms= track_features[0]['duration_ms']
#     time_signature= track_features[0]['time_signature']
    track_data = [id, name, album, artist, danceability,energy,loudness,speechiness, acousticness, instrumentalness,
                  liveness, valence, tempo, duration_ms]
except:
    
    track_data = [id,name, album, artist, 'null','null','null','null', 'null', 'null',
                  'null', 'null','null', 'null']


# In[103]:


track_data


# In[104]:


track_data = pd.DataFrame(track_data).T.values.tolist() #Transpose dataframe to list


# In[105]:


features = pd.DataFrame(track_data, columns = ['id','name', 'album', 'artist', 'danceability','energy','loudness','speechiness',
                                               'acousticness','instrumentalness','liveness','valence', 'tempo', 'duration_ms'
                                               ])


# In[106]:


features


# Check against Billboard Hot100

# In[107]:


user_input_name = features['name'][0]

# select everything before brackets


# In[108]:


user_input_name=re.sub(" \(.*?\)","",user_input_name)


# In[75]:


user_input_name


# In[76]:


url = "https://www.billboard.com/charts/hot-100"


# In[77]:


# 3. download html with a get request
response = requests.get(url)
response.status_code # 200 status code means OK!


# In[78]:


# 4.1. parse html (create the 'soup')
soup = BeautifulSoup(response.content, "html.parser")


# In[79]:


hot_songs = []
# artist = []
num_iter = len(soup.select('span.chart-element__information span'))

for i in range(0,num_iter,7): # start at position 0, iterate through the len of the column, stop at 7
    hot_songs.append(soup.select('span.chart-element__information span')[i].get_text())
#     artist.append(soup.select('span.chart-element__information span')[i+1].get_text()) # artist is at position 0+1


# In[80]:


hot_songs = pd.DataFrame(hot_songs)


# In[109]:


# hot_songs[0]=hot_songs[0].str.extract(" \(.*?\)")

hot_songs[0] = hot_songs[0].str.replace(" \(.*?\)","", regex = True)


# In[82]:


hot_songs_x = list(hot_songs[0])


# After that, you want to send the Spotify audio features of the 
# submitted song to the clustering model, which should return a cluster number.

# In[83]:


# Change object to float

features['danceability'] = pd.to_numeric(features['danceability'],errors='coerce')
features['energy'] = pd.to_numeric(features['energy'],errors='coerce')
features['loudness'] = pd.to_numeric(features['loudness'],errors='coerce')
features['speechiness'] = pd.to_numeric(features['speechiness'],errors='coerce')
features['acousticness'] = pd.to_numeric(features['acousticness'],errors='coerce')
features['instrumentalness'] = pd.to_numeric(features['instrumentalness'],errors='coerce')
features['liveness'] = pd.to_numeric(features['liveness'],errors='coerce')
features['valence'] = pd.to_numeric(features['valence'],errors='coerce')
features['tempo'] = pd.to_numeric(features['tempo'],errors='coerce')
features['duration_ms'] = pd.to_numeric(features['duration_ms'],errors='coerce')


# In[84]:


audio_features_less = features.drop(['id','name','album','artist'],axis=1) 


# In[85]:


audio_features_less = pd.DataFrame(audio_features_less, index = [0])
audio_features_less


# In[86]:


X_prep_less = scaler.transform(audio_features_less)


# In[87]:


X_prep_less


# In[88]:


if user_input_name in hot_songs_x:
    print("May I suggest the following hot song:")
    print(random.choice(hot_songs))
                 
else:
    
    try:
        cluster = kmeans.predict(X_prep_less)
        narrowed_down = database_songs[(database_songs['clusters'] == cluster[0])]
        suggestion_list = narrowed_down['id'].tolist()
        selection = random.choice(suggestion_list)
        suggestion = database_songs[(database_songs['id']==selection)]
        output = suggestion[["name", "artist","url"]]
        print("May I suggest the following song:")
        for col_name in output.columns: 
            print(col_name+':', output[col_name]. value_counts(). idxmax())
        
    except:
        print("no recommendation")  


# In[97]:


# cluster = kmeans.predict(X_prep_less)
# cluster

