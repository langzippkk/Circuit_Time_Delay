from preprocessing import preprocessing_data
from data_generating import data_generating
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from utils import generating_tensor_X,generating_tensor_Y,generating_tensor_X_test,generating_CNN_intput,\
test_train,generating_tensor_X_SVM
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


if __name__ == "__main__":
	# file = 'data.json'
	# data = data_generating(file)
	# X_init,Y_init =data.generate_short_X()
	# X = data.generate_alternating_df(2)
	# X.index = [i for i in range(len(X))]
	# X = X.drop(X.index[0])
	# X.to_pickle("Data.pkl")
	# Y = pd.DataFrame(Y_init)
	# Y.to_pickle("label724")
	##### save the initial data ##############
	X = pd.read_pickle('Data.pkl')
	Y = pd.read_pickle("label724")
	## train test split
	split_Y = int(len(Y)*0.8)
	split = split_Y*10
	X_train = X[:split ]
	X_test = X[split :]
	Y_train = Y[:split_Y]
	Y_test = Y[split_Y:]
	preprocess = preprocessing_data(X,0.8)
	X_train = preprocess.filter_columns(X_train)
	X_train = preprocess.normalise(X_train)
	X_train = preprocess.categorical(X_train)
	X_train = preprocess.location_transform(X_train)
	X_test = preprocess.filter_columns(X_test)
	X_test = preprocess.normalise(X_test)
	X_test = preprocess.categorical(X_test)
	X_test = preprocess.location_transform(X_test)
	X_test.index = [i for i in range(1,len(X_test)+1)]
	Y_test.index = [i for i in range(len(Y_test))]
	SVM_X_Train = generating_tensor_X_SVM(X_train)
	SVM_X_Test = generating_tensor_X_SVM(X_test)
	Y_train = generating_tensor_Y(Y_train)
	Y_test =  generating_tensor_Y(Y_test)
	parameters = {
    "kernel": ["rbf"],
    "C": [1,10,10,100,1000],
    "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }
	grid = GridSearchCV(SVR(), parameters, cv=5, verbose=2)
	grid.fit(SVM_X_Train,Y_train)
	C = grid.best_params_['C']
	epsilon =  grid.best_params_['gamma']
	kf = KFold(n_splits=2)
	score = []
	for train_index, test_index in kf.split(SVM_X_Train):
	    print("TRAIN:", train_index, "TEST:", test_index)
	    X_train, X_test = np.array(SVM_X_Train)[train_index], np.array(SVM_X_Train)[test_index]
	    y_train, y_test = np.array(Y_train)[train_index], np.array(Y_train)[test_index]
	    regr = make_pipeline(StandardScaler(), SVR(C=C, epsilon=epsilon))
	    regr.fit(X_train,y_train)
	    predicted = regr.predict(X_test)
	    score.append(mean_squared_error(predicted,y_test))
	print(np.mean(score))
	## testing
	regr = make_pipeline(StandardScaler(), SVR(C=C, epsilon=epsilon))
	regr.fit(SVM_X_Train,Y_train)
	predicted = regr.predict(SVM_X_Test)
	print("The testing MSE with tuned parameters is"+str(mean_squared_error(predicted,Y_test)))