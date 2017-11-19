import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

def preparing_dataset():
	players = pd.read_csv('Datasets/Players.csv', index_col=0)
	stats = pd.read_csv('Datasets/Seasons_Stats.csv', index_col=0)

	data = pd.merge(stats, players[['Player', 'height', 'weight']], left_on='Player', right_on='Player', how='left')
	data = data.fillna(value=0)
	data = data.drop(['blanl', 'blank2', 'Tm', 'Age', 'G', 'GS'], axis='columns')
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
	
def predict_neural_network():	
	columns_count = X_train.shape[1]
	seq.add(Dense(40, activation='relu', input_dim=columns_count))
	seq.add(Dropout(0.5))
	seq.add(Dense(30, activation='relu'))
	seq.add(Dropout(0.5))
	seq.add(Dense(len(encoder.classes_), activation='softmax'))
	seq.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	seq.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0, verbose=1)
	pred = seq.predict(X_test)
	return pred
	
def note_results(results, path):
	test_stats =  data[test_stats_filter]
	idx = test_stats['Player'].values
	real = test_stats['Pos'].values
	results_str = encoder.inverse_transform(results)
	test_result_data = pd.DataFrame(index=idx, data={'Real': real, 'Predicted': results_str})
	test_result_data.to_csv(path, sep=',')
	count = len(test_result_data)
	hit_count = len(test_result_data[test_result_data['Real'] == test_result_data['Predicted']])
	print ("Od ukupno", count, " pogodjeno je", hit_count, ", sto je ", accuracy_score(real,results_str)*100, "%")

	miss = test_result_data[test_result_data['Real'] != test_result_data['Predicted']]
	hit_neighbors = hit_count;
	for idx in miss.index:
		[pos1, pos2] = miss.loc[idx]
		if(is_heighbors(pos1,pos2)):
			hit_neighbors=hit_neighbors+1
	print ("Od ukupno", count, ", ako se susedne pozicije uzmu u obzir, pogodjeno je", hit_neighbors, ", sto je", (hit_neighbors/count)*100, "%")	
	
def is_heighbors (pos1, pos2):
	if pos1=='PG':
		if pos2== 'SG':
			return True
	if pos1=='SG':
		if (pos2=='PG') | (pos2 =='SF'):
			return True
	if pos1=='SF':
		if (pos2=='SG') | (pos2 =='PF'):
			return True
	if pos1=='PF':
		if (pos2=='SF') | (pos2 =='C'):
			return True
	if pos1=='C':
		if pos2 == 'PF':
			return True
	return False
	
def predict_KNN ():
	#k = get_best_k()
	KNN.n_neighbors=3
	KNN.fit(X_train,y_train)	
	pred = KNN.predict(X_test)
	return pred
	
def get_best_k():
	neighbors = range(1,2,50)
	cv_scores = []
	
	for k in neighbors:
		KNN = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score (KNN, X_train, y_train, cv=10, scoring='accuracy')
		cv_scores.append(scores.mean())
	return optimal_k
	
def predict_RFC ():
	RFC.fit(X_train,y_train)
	pred = RFC.predict(X_test)
	return pred

def predict_SVC():
	SVC.fit(X_train, y_train)
	SVC.score(X_train, y_train)
	pred = model.predict(X_test)
	return pred
	
	
####################################################################################

SVC = svm.SVC()
KNN = KNeighborsClassifier()
RFC = RandomForestClassifier()
seq = Sequential()
encoder = LabelBinarizer()
scaler = StandardScaler()

data = preparing_dataset();
data = remove_by_minutes(data)
data = remove_by_year(data)
data = correct_FG_percentage(data)
data = correct_FT_percentage(data)
data = prepare_totals(data)
data.reset_index(inplace=True, drop=True)


X = data.drop(['Player', 'Pos', 'MP'], axis=1).as_matrix()
y = data['Pos'].as_matrix()
y_encoded = encoder.fit_transform(y)
X_scaled = scaler.fit_transform(X)

test_stats_filter = data['Year'] == 2017
X_train = X_scaled[~test_stats_filter]
y_train = y_encoded[~test_stats_filter]

X_test = X_scaled[test_stats_filter]
y_test = y_encoded[test_stats_filter]
"""
#################### NEURAL NETWORK ########################
neural_network_results = predict_neural_network()
print ('\n------- Neural network results --------')
note_results(neural_network_results,'Neural_network_results.csv')
############################################################

######################### KNN ##############################
KNN_results = predict_KNN()
print('\n------------- KNN results ---------------')
note_results(KNN_results, 'KNN_results.csv')
############################################################
"""
#################### Random Forest #########################
RFC_results = predict_RFC()
print('\n-------------- RFC results -------------')
note_results(RFC_results, 'RFC_results.csv')
############################################################

####################### SVC ################################
#SVC_results = predict_SVC()
#print('\n-------------- SVC results -------------')
#note_results(SVC_results, 'SVC_results.csv')
############################################################
