from tensorflow.python.keras.layers import Input, Dense,LSTM
from keras.initializers import glorot_uniform, glorot_normal, RandomUniform
from keras.callbacks import History 
from keras.models import load_model
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from utils import generating_tensor_X,generating_tensor_Y,generating_tensor_X_test,generating_CNN_intput,\
test_train
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate,Reshape
from model import train_CNN
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
	X_train,Y_train,X_test,Y_test = test_train(X,Y)
	CNN_train = generating_CNN_intput(X_train)
	CNN_test = generating_CNN_intput(X_test)

	## parameter tuning
	parameter_dict={'epochs':30,'batch':[128,64,32],'lr':[0.0005,0.0006,0.0007],'dropout':[0.1,0.2,0.3]}
	epochs =  parameter_dict['epochs']
	tuning_result = 100
	tuning_para = [0]*3
	for batch_size in parameter_dict['batch']:
	    for lr in parameter_dict['lr']:
	        for drop_out in parameter_dict['dropout']:
	            res = train_CNN(CNN_train,Y_train,CNN_test,Y_test,batch_size,drop_out,lr,epochs)
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
	def create_CNN(serie_size,n_features,drop_out):
	    inputs = Input(shape=(serie_size,n_features))     
	    x =(Conv1D(64,kernel_size=2,input_shape=(serie_size,n_features),padding='same',activation='relu'))(inputs)
	    x = (Dropout(drop_out))(x)
	    x =(Conv1D(32,kernel_size=2,padding='same',activation='relu'))(x)
	    x = (MaxPooling1D(pool_size=2))(x)
	    x = (Dropout(drop_out))(x)
	    model = Model(inputs,x)
	    return model
	temp_1 = create_CNN(2,41,drop_out)
	temp_3 = create_CNN(2,41,drop_out)
	temp_2 = create_CNN(2,41,drop_out)
	temp_4 = create_CNN(2,41,drop_out)
	optimizer = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
	init = glorot_normal(seed=None)
	init1 = RandomUniform(minval=-0.05, maxval=0.05)
	## 10 * 32 
	combinedInput = concatenate([temp_1.output,temp_2.output,temp_3.output,temp_4.output])
	combinedInput = Reshape((4,32), input_shape=(4*32,))(combinedInput)
	lstm = LSTM(units=16, dropout=drop_out, recurrent_dropout=drop_out,return_sequences=False, kernel_initializer=init)
	output = lstm(combinedInput)
	output = Dense(1, activation='linear', kernel_initializer= init1)(output)
	model = Model(inputs=[temp_1.input,temp_2.input,temp_3.input,temp_4.input], outputs=output)
	model.compile(loss='mean_squared_error',optimizer=optimizer)
	model.summary()
	y_train = np.asarray(Y_train)
	train_history = model.fit(x=[i for i in CNN_train],y=y_train,epochs=epochs,batch_size=batch_size)
	model.save('CNN_LSTM.h5') 
	model = load_model('CNN_LSTM.h5')
	predicted = model.predict(CNN_test,batch_size=1)
	print("The testing MSE with tuned parameters is"+str(mean_squared_error(predicted,Y_test)))