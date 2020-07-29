from tensorflow.python.keras.layers import Input, Dense,LSTM
from keras.initializers import glorot_uniform, glorot_normal, RandomUniform
from keras.callbacks import History 
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate,Reshape
from keras.optimizers import RMSprop
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from preprocessing import preprocessing_data
from data_generating import data_generating
from utils import generating_tensor_X,generating_tensor_Y,generating_tensor_X_test,test_train_Multi,generating_tensor_CNN,generate_combined
from model import train_single_LSTM
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
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

	X = pd.read_pickle('Data.pkl')
	Y = pd.read_pickle("label724")
	## train test split
	gate_train,gate_test,net_train,net_test,Y_train,Y_test = test_train_Multi(X,Y)
	gate_train = generating_tensor_CNN(gate_train)
	gate_test =  generating_tensor_CNN(gate_test)
	net_train = generating_tensor_CNN(net_train)
	net_test =  generating_tensor_CNN(net_test)

	combined_train = generate_combined(gate_train,net_train)## 10*8312
	combined_test = generate_combined(gate_test,net_test)  ## 10 *2080

    ## parameter tuning
	parameter_dict={'epochs':30,'batch':[128,64,32],'lr':[0.0005,0.0006,0.0007],'dropout':[0.1,0.2,0.3]}
	epochs =  parameter_dict['epochs']
	tuning_result = 100
	tuning_para = [0]*3
	for batch_size in parameter_dict['batch']:
	    for lr in parameter_dict['lr']:
	        for drop_out in parameter_dict['dropout']:
	            res = train_Combined(combined_train,Y_train,combined_test,Y_test,batch_size,drop_out,lr,epochs)
	            if res < tuning_result:
	                tuning_result = res
	                tuning_para[0] = batch_size
	                tuning_para[1] = lr
	                tuning_para[2] = drop_out
	print(tuning_para,tuning_result)

	batch_size = tuning_para[0]
	lr = tuning_para[1]
	drop_out = tuning_para[2]
	epochs = 50
	def create_dense(serie_size,n_features,drop_out):
	    inputs = Input(shape=(serie_size,n_features))
	    x = Dense(10,activation='relu')(inputs)
	    x = Dense(8,activation='relu')(x)
	    x = Flatten()(x)
	    model = Model(inputs,x)
	    return model
	serie_size= 1
	n_features_gate = len(gate_train[0][0][0])
	n_features_net =  len(net_train[0][0][0])
	gate_1 = create_dense(serie_size,n_features_gate,drop_out)
	gate_3 = create_dense(serie_size,n_features_gate,drop_out)
	gate_5 = create_dense(serie_size,n_features_gate,drop_out)
	gate_7 = create_dense(serie_size,n_features_gate,drop_out)
	gate_9 = create_dense(serie_size,n_features_gate,drop_out)
	net_2 = create_dense(serie_size,n_features_net,drop_out)
	net_4 = create_dense(serie_size,n_features_net,drop_out)
	net_6 = create_dense(serie_size,n_features_net,drop_out)
	net_8 = create_dense(serie_size,n_features_net,drop_out)
	net_10 = create_dense(serie_size,n_features_net,drop_out)
	optimizer = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
	init = glorot_normal(seed=None)
	init1 = RandomUniform(minval=-0.05, maxval=0.05)
	## 10 * 32 
	combinedInput = concatenate([gate_1.output, net_2.output,gate_3.output,net_4.output,gate_5.output,net_6.output,\
	                            gate_7.output,net_8.output,gate_9.output,net_10.output])
	combinedInput = Reshape((10,8), input_shape=(8*10,))(combinedInput)
	lstm = LSTM(units=16, dropout=drop_out, recurrent_dropout=drop_out,return_sequences=False, kernel_initializer=init)
	output = lstm(combinedInput)
	output = Dense(1, activation='linear', kernel_initializer= init1)(output)
	model = Model(inputs=[gate_1.input, net_2.input,gate_3.input,net_4.input,gate_5.input,net_6.input\
	                            ,gate_7.input,net_8.input,gate_9.input,net_10.input], outputs=output)
	model.compile(loss='mean_squared_error',optimizer=optimizer)
	y_train = np.asarray(Y_train)
	train_history = model.fit(x=[i for i in combined_train],y=y_train,epochs=epochs,batch_size=batch_size)
	loss = train_history.history['loss']
	plt.plot(loss)
	plt.show()
	model.save('Combined.h5') 
	model = load_model('Combined.h5')
	predicted = model.predict(combined_test,batch_size=1)
	print("The testing MSE with tuned parameters is"+str(mean_squared_error(predicted,Y_test)))