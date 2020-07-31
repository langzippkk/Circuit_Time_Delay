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
from preprocessing import preprocessing_data
from data_generating import data_generating
from utils import generating_tensor_X,generating_tensor_Y,test_train
from model import train_single_LSTM
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
if __name__ == "__main__":
    '''
    This main function tunes hyperparameter by using the model in models.py and find the 
    best hyperparameters to train a model.
    Output: print the testing MSE for the best model

    '''

    # uncomment it if read directly from json
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
    X_train,Y_train,X_test,Y_test = test_train(X,Y)

    ## parameter tuning 
    parameter_dict={'epochs':30,'batch':[128,64,32],'lr':[0.0007,0.0008,0.001],'dropout':[0.1,0.2,0.3]}
    epochs =  parameter_dict['epochs']
    tuning_result = 100
    tuning_para = [0]*3
    for batch_size in parameter_dict['batch']:
        for lr in parameter_dict['lr']:
            for drop_out in parameter_dict['dropout']:
                res = train_single_LSTM(X_train,Y_train,X_test,Y_test,batch_size,drop_out,lr,epochs)
                if res < tuning_result:
                    tuning_result = res
                    tuning_para[0] = batch_size
                    tuning_para[1] = lr
                    tuning_para[2] = drop_out
    ## testing with tuned parameters
    batch_size = tuning_para[0]
    lr = tuning_para[1]
    drop_out = tuning_para[2]
    epochs = 50
    serie_size= len(X_train[0]) # 4
    n_features =len(X_train[0][0]) # 82
    lstm_model  = Sequential()
    optimizer = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
    init = glorot_normal(seed=None)
    init1 = RandomUniform(minval=-0.05, maxval=0.05)
    lstm_model.add(LSTM(units=128, dropout=drop_out, recurrent_dropout=drop_out,input_shape=(serie_size,n_features),return_sequences=True, kernel_initializer=init))
    lstm_model.add(LSTM(units=64, dropout=drop_out, recurrent_dropout=drop_out,return_sequences=False, kernel_initializer=init))
    lstm_model.add(Dense(1, activation='linear', kernel_initializer= init1))
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    lstm_model.summary()
    x_train = np.asarray(X_train)
    y_train = np.asarray(Y_train)
    x_test = np.asarray(X_test)
    train_history = lstm_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,verbose=0,shuffle=True) 
    loss = train_history.history['loss']
    plt.plot(loss)
    plt.show()
    #         lstm_model.reset_states()
    lstm_model.save('single_LSTM.h5') 
    model = load_model('single_LSTM.h5')
    predicted = model.predict(x_test,batch_size=1)
    print("The testing MSE with tuned parameters is"+str(mean_squared_error(predicted,Y_test)))