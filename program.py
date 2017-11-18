import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

def preparing_dataset():
	players = pd.read_csv('Datasets/Players.csv', index_col=0)
	stats = pd.read_csv('Datasets/Seasons_Stats.csv', index_col=0)

	data = pd.merge(stats, players[['Player', 'height', 'weight']], left_on='Player', right_on='Player', how='left')
	data = data.fillna(value=0)
	data = data.drop(['Tm', 'Age', 'G', 'GS'], axis='columns')
	#Removing all rows on year change in dataset (there was row with just ID)
	filter = data['Player'] != 0
	data = data[filter]
	filter = (data['Pos']=="PG") | (data['Pos']=="SG") | (data['Pos']=="SF") | (data['Pos']=="PF") | (data['Pos']=="C")
	data = data[filter]
	
	#Removing a star sign at the end of the names of some players
	data['Player'] = data['Player'].str.replace('*','')
	return data

def remove_by_minutes(data, minutes=500):
	filter = data['MP']>=minutes; 
	data = data[filter]
	return data;

def remove_by_year(data,year=1980):
	filter = data['Year'] >= year
	data = data[filter]
	return data;
	
def correct_FG_percentage (data, field_goals_attempts=100):
	data.loc[data['FGA'] <= field_goals_attempts,'FG%'] = data['FG%'].mean()
	return data
	
def correct_FT_percentage (data, free_throws_attempts=20):
	data.loc[data['FTA'] <= free_throws_attempts,'FT%'] = data['FT%'].mean()
	return data
	
def prepare_totals(data,minutes_to_mull=36):
	totals = ['PER', 'OWS', 'DWS', 'WS', 'OBPM', 'DBPM', 'BPM', 'VORP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
	for col in totals:
		data[col] = minutes_to_mull*data[col]/data['MP']
	return data;
	

def train(X_train, y_train):	
	columns_count = X_train.shape[1]
	model.add(Dense(40, activation='relu', input_dim=columns_count))
	model.add(Dropout(0.5))
	model.add(Dense(30, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(len(encoder.classes_), activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0, verbose=1)
	return model
	
	
def test_by_seasion():
	test_stats =  data[test_stats_filter]
	test_data = X_scaled[test_stats_filter]
	idx = test_stats['Player'].values
	real = test_stats['Pos'].values
	predicted = encoder.inverse_transform(model.predict(test_data))
	test_result_data = pd.DataFrame(index=idx, data={'Real': real, 'Predicted': predicted})
	count = len(test_result_data)
	hit = len(test_result_data[test_result_data['Real'] == test_result_data['Predicted']])
	print (test_result_data)
	print ("Od ukupno ", count, " pogodjeno je", hit, ", sto je ", (hit/count)*100, "%")

####################################################################################

model = Sequential()
encoder = LabelBinarizer()
scaler = StandardScaler()
data = preparing_dataset();
data = remove_by_minutes(data)
data = remove_by_year(data)
data = correct_FG_percentage(data)
data = correct_FT_percentage(data)
data = prepare_totals(data)
data.reset_index(inplace=True, drop=True)
#exit()

X = data.drop(['Player', 'Pos', 'MP'], axis=1).as_matrix()
y = data['Pos'].as_matrix()
y_encoded = encoder.fit_transform(y)
X_scaled = scaler.fit_transform(X)

test_stats_filter = data['Year'] == 2017
X_train = X_scaled[~test_stats_filter]
y_train = y_encoded[~test_stats_filter]
print (X_train)
print (y_train)

train(X_train, y_train)

test_by_seasion()



