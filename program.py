import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


#Reading players and stats from CSVs
players = pd.read_csv('Datasets/Players.csv', index_col=0)
stats = pd.read_csv('Datasets/Seasons_Stats.csv', index_col=0)

#Merging players height and weight with stats by player name
all_data = pd.merge(stats, players[['Player', 'height', 'weight']],
left_on='Player', right_on='Player', right_index=False, 
how='left', sort=False).fillna(value=0)

#Removing team column from stats becouse it is not necessary
data = all_data.drop(['Tm'], axis=1)

#Removing all rows on year change in dataset (there was row with just ID)
filter = data['Player'] != 0
data = data[filter]

#Removing all rows with bad formated possition column (valid possitions are PG, SG, SF, PF and C)
filter = (data['Pos']=="PG") | (data['Pos']=="SG") | (data['Pos']=="SF") | (data['Pos']=="PF") | (data['Pos']=="C")
data = data[filter] 

#Removing a star sign at the end of the names of some players
data['Player'] = data['Player'].str.replace('*','')

####################################################################################

#Removing all players which plays less then 300 min per seasion
#Players with less then 300 MP are not good indicator for this project
filter = data['MP']>300; 
data = data[filter]

#Removing all players which plays earlyer then 1990s
#Basketball was different
filter = data['Year'] > 1990
data = data[filter]

#Some of stats presenting sum of all matches for one seasion
#All of these stats but per 36 min on court is better indicator
totals = ['PER', 'OWS', 'DWS', 'WS', 'OBPM', 'DBPM', 'BPM', 'VORP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
for col in totals:
	data[col] = 36*data[col]/data['MP']

####################################################################################

#Index reset becouse of some row removing
data.reset_index(inplace=True, drop=True)


print(data)
