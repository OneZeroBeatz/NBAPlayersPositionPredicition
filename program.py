import pandas as pd
import numpy as np
#F measure - dodato
#PCA 
#SEQ kfold - DODATO

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder, OneHotEncoder

from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold


import matplotlib.pyplot as plt


def preparing_dataset():
	players = pd.read_csv('Datasets/Players.csv', index_col=0)
	stats = pd.read_csv('Datasets/Seasons_Stats.csv', index_col=0)

	data = pd.merge(stats, players[['Player', 'height', 'weight']], left_on='Player', right_on='Player', how='left')
	data = data.fillna(value=0)
	data = data.drop(['blanl', 'blank2', 'Tm', 'Age'], axis='columns')

	#Removing all rows on year change in dataset (there was row with just ID)
	data = data[data['Player'] != 0]
	filter = (data['Pos']=="PG") | (data['Pos']=="SG") | (data['Pos']=="SF") | (data['Pos']=="PF") | (data['Pos']=="C")
	data = data[filter]
	
	#Removing a star sign at the end of the names of some players
	data['Player'] = data['Player'].str.replace('*','')
	return data
#60-75 (71) za 1980
def remove_by_minutes(data, minutes=71): 
	filter = data['MP']>=minutes; 
	data = data[filter]
	return data;

def remove_by_year(data,year=1980):
	filter = data['Year'] >= year
	data = data[filter]
	return data;
	
def correct_FG_percentage (data, field_goals_attempts=30):
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

def note_results(results, path, encoder):

	test_stats_filter = data['Year'] == test_seasion
	test_stats =  data[test_stats_filter]
	idx = test_stats['Player'].values
	real = test_stats['Pos'].values
	results_str = encoder.inverse_transform(results)
	test_result_data = pd.DataFrame(index=idx, data={'Real': real, 'Predicted': results_str})
	test_result_data.to_csv(path, sep=',')
	count = len(test_result_data)
	hit_count = len(test_result_data[test_result_data['Real'] == test_result_data['Predicted']])
	print ("Redovni: \t", hit_count, "/", count, "(", round (accuracy_score(real,results_str)*100,3), "%)")

	miss = test_result_data[test_result_data['Real'] != test_result_data['Predicted']]
	hit_neighbors = hit_count;
	for idx in miss.index:
		[pos1, pos2] = miss.loc[idx]
		if(is_heighbors(pos1,pos2)):
			hit_neighbors=hit_neighbors+1
	print ("Susedni:\t", hit_neighbors, "/",count , "(", round((hit_neighbors/count)*100,3), "%)")	
	return 100*accuracy_score(real,results_str)
	
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
		
def get_best_k():
	neighbors = range(1,2,50)
	cv_scores = []
	
	for k in neighbors:
		KNN = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score (KNN, X_train, y_train, cv=10, scoring='accuracy')
		cv_scores.append(scores.mean())
	return optimal_k

def predict(model, X_train, y_train, X_test):
	model.fit(X_train,y_train)
	model_results = model.predict(X_test)
	print('Test data score', round(accuracy_score(y_train,model.predict(X_train))*100,3))
		
	return model_results

def dimensionality_reduction(data):
	data = data.drop(['MP','G'], axis=1)
	return data
######################################
	
def split_sets(encoder):
	X = data.drop(['Pos', 'Player'], axis=1).as_matrix()
	y = data['Pos'].as_matrix()
	y_encoded = encoder.fit_transform(y)
	X_scaled = scaler.fit_transform(X)
	
	test_stats_filter = data['Year'] == test_seasion
	X_train = X_scaled[~test_stats_filter]
	y_train = y_encoded[~test_stats_filter]
	
	X_test = X_scaled[test_stats_filter]
	y_test = y_encoded[test_stats_filter]
	return [X_train, y_train, X_test, y_test]
	
def test (model, encoder, X_train, y_train, X_test, path):
	model_results = predict(model, X_train, y_train, X_test)
	note_results(model_results,path, encoder)

def test_SVC(encoder):
	X_train, y_train, X_test, y_test = split_sets(encoder)
	print('\n------------------- SVC results -----------------------')
	k_fold_cross_validation(SVC, encoder, X_train, y_train)
	test(SVC, encoder, X_train, y_train, X_test, 'SVC_results.csv')	
	
def test_KNN(encoder):
	X_train, y_train, X_test, y_test = split_sets(encoder)
	print('\n------------------- KNN results -----------------------')
	k_fold_cross_validation(KNN, encoder, X_train, y_train)
	test(KNN, encoder, X_train, y_train, X_test, 'KNN_results.csv')
	
def test_naive_bayes(encoder):
	X_train, y_train, X_test, y_test = split_sets(encoder)
	print('\n---------------- Naive Bayes results ------------------')	
	k_fold_cross_validation(naive_bayes, encoder, X_train, y_train)
	test(naive_bayes, encoder, X_train, y_train, X_test, 'naive_bayes_results.csv')

def test_LDA(encoder):
	X_train, y_train, X_test, y_test = split_sets(encoder)
	print('\n-------------------- LDA results ----------------------')
	k_fold_cross_validation(LDA, encoder, X_train, y_train)
	test(LDA, encoder, X_train, y_train, X_test, 'LDA_results.csv')
	
def test_DTC(encoder):
	X_train, y_train, X_test, y_test = split_sets(encoder)
	print('\n-------------------- DTC results ----------------------')
	k_fold_cross_validation(DTC, encoder, X_train, y_train)
	test(DTC, encoder, X_train, y_train, X_test, 'DTC_results.csv')

def test_RFC(encoder):
	X_train, y_train, X_test, y_test = split_sets(encoder)
	print('\n-------------------- RFC results ----------------------')
	k_fold_cross_validation(RFC, encoder, X_train, y_train)
	test(RFC, encoder, X_train, y_train, X_test, 'RFC_results.csv')	
	
def test_SEQ(encoder):	
	X_train, y_train, X_test, y_test = split_sets(encoder)
	columns_count = X_train.shape[1]
	SEQ.add(Dense(40, activation='relu', input_dim=columns_count))
	SEQ.add(Dropout(0.5))
	SEQ.add(Dense(30, activation='relu'))
	SEQ.add(Dropout(0.5))
	SEQ.add(Dense(len(encoder.classes_), activation='softmax'))
	SEQ.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	
	k_fold_cross_validation(SEQ, encoder, X_train, y_train)
	
	SEQ.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0, verbose=1)
	SEQ_results = SEQ.predict(X_test)
	print('\n-------------------- SEQ results ----------------------')
	print('Test data score', round(accuracy_score(encoder.inverse_transform(y_train),encoder.inverse_transform(SEQ.predict(X_train)))))
	note_results(SEQ_results,'SEQ_results.csv', encoder)
	
def k_fold_cross_validation(model,encoder,X, Y, k = 10):
	k_fold = KFold(n_splits = k, random_state=None, shuffle=False)
	#for train_index, val_index in k_fold.split(X):
	#	X_train, X_val = X[train_index], X[val_index]
	#	y_train, y_val = Y[train_index], Y[val_index]
	#	if (model == SEQ):			
	#		model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split =0, verbose= 1)
	#		predict_results = model.predict(X_val)
	#		predict_results = encoder.inverse_transform(predict_results)
	#		y_val = encoder.inverse_transform(y_val)
	#		print('Test data score', round(accuracy_score(encoder.inverse_transform(y_train),encoder.inverse_transform(model.predict(X_train)))))
	#	if (model != SEQ):
	#		predict_results = predict(model,X_train,y_train, X_val)
	#	print('Accuracy score', round(accuracy_score(y_val,predict_results)*100,3))
	#	print('F1 measure    ', round(f1_score(y_val,predict_results, average ='weighted')*100,3))
	#	#print('Classif report', classification_report(y_val,predict_results)*100,3)
	#	#print('Confusion matr\n', confusion_matrix(y_val,predict_results))
	#	print('\n')

		
	
####################################################################################

test_seasion = 2017

SVC = SVC()
KNN = KNeighborsClassifier()
naive_bayes = GaussianNB()
LDA = LinearDiscriminantAnalysis()
DTC = DecisionTreeClassifier()
RFC = RandomForestClassifier()

SEQ = Sequential()

binarizer = LabelBinarizer()
encoder = LabelEncoder()
scaler = StandardScaler()

data = preparing_dataset();
data = remove_by_minutes(data)
data = remove_by_year(data)
data = correct_FG_percentage(data)
data = correct_FT_percentage(data)
data = prepare_totals(data)
data = dimensionality_reduction(data)
data.reset_index(inplace=True, drop=True)




#ENCODER
#test_SVC(encoder)
#test_KNN(encoder)
#test_naive_bayes(encoder)
#test_LDA(encoder)
#test_DTC(encoder)
#test_RFC(encoder)

#BINARIZER
#test_SEQ(binarizer)


