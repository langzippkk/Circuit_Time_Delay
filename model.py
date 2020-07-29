from keras.models import load_model
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
# from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.python.keras.layers import Input,Dense,LSTM
from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers import LSTM
from tensorflow.keras.layers import concatenate,Reshape
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform, glorot_normal, RandomUniform
from keras.callbacks import History 
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import numpy as np

def train_single_LSTM(X_train,Y_train,X_test,Y_test,batch_size,drop_out,lr,epochs):
    serie_size= len(X_train[0]) # 4
    n_features =len(X_train[0][0]) # 82
#     epochs = parameter_dict['epochs']
#     batch =  parameter_dict['batch']
#     lr =  parameter_dict['lr']
    ## input shape : samples, time steps, and features.
    lstm_model  = Sequential()
    optimizer = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
    #optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)
    init = glorot_normal(seed=None)
    init1 = RandomUniform(minval=-0.05, maxval=0.05)
    lstm_model.add(LSTM(units=128, dropout=drop_out, recurrent_dropout=drop_out,input_shape=(serie_size,n_features),return_sequences=True, kernel_initializer=init))
    lstm_model.add(LSTM(units=64, dropout=drop_out, recurrent_dropout=drop_out,return_sequences=False, kernel_initializer=init))
    lstm_model.add(Dense(1, activation='linear', kernel_initializer= init1))
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    lstm_model.summary()
    kf = KFold(n_splits=3)
    score = []
    for train_index, test_index in kf.split(X_train):
        x_train, x_valid = np.array(X_train)[train_index], np.array(X_train)[test_index]
        y_train, y_valid = np.array(Y_train)[train_index], np.array(Y_train)[test_index]
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_valid = np.asarray(x_valid)
        y_valid = np.asarray(y_valid)
        x_test = np.asarray(X_test)
        train_history = lstm_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_valid, y_valid), verbose=0,shuffle=True) 
        loss = train_history.history['loss']
        val_loss = train_history.history['val_loss']
        plt.plot(loss)
        plt.plot(val_loss)
        plt.legend(['loss', 'val_loss'])
        plt.show()
#         lstm_model.reset_states()
        lstm_model.save('my_model.h5') 
        model = load_model('my_model.h5')
        x_test = np.asarray(X_test)
        #predicted = model.predict(x_test,batch_size=1)
        score.append(val_loss[-1])
    return np.mean(score)



def train_CNN(combined_train,Y_train,combined_test,Y_test,batch_size,drop_out,lr,epochs):
    def create_CNN(serie_size,n_features,drop_out):
        inputs = Input(shape=(serie_size,n_features))     
        x =(Conv1D(64,kernel_size=2,input_shape=(serie_size,n_features),padding='same',activation='relu'))(inputs)
        x = (Dropout(drop_out))(x)
        x =(Conv1D(32,kernel_size=2,padding='same',activation='relu'))(x)
        x = (MaxPooling1D(pool_size=2))(x)
        x = (Dropout(drop_out))(x)
        model = Model(inputs,x)
        return model
    kf = KFold(n_splits=3)
    score = []
    for train_index, test_index in kf.split(combined_train[0]):
        x_train, x_valid = [i[train_index] for i in combined_train], [i[test_index] for i in combined_train]
        print(len(x_train))
        print(len(x_train[0][0]))
        y_train, y_valid = np.array(Y_train)[train_index], np.array(Y_train)[test_index]
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
        train_history = model.fit(x=[i for i in x_train],y=y_train,epochs=epochs,validation_data=(x_valid, y_valid),batch_size=batch_size)
        val_loss = train_history.history['val_loss'][-1]
        score.append(val_loss)
    return(np.mean(score))



def train_Combined(combined_train,Y_train,combined_test,Y_test,batch_size,drop_out,lr,epochs):
    def create_dense(serie_size,n_features,drop_out):
        inputs = Input(shape=(serie_size,n_features))
        x = Dense(10,activation='relu')(inputs)
        x = Dense(8,activation='relu')(x)
        x = Flatten()(x)
        model = Model(inputs,x)
        return model
    kf = KFold(n_splits=3)
    score = []
    for train_index, test_index in kf.split(combined_train[0]):
        print(train_index)
        x_train, x_valid = [i[train_index] for i in combined_train], [i[test_index] for i in combined_train]
        y_train, y_valid = np.array(Y_train)[train_index], np.array(Y_train)[test_index]
        
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
        print(gate_1.output)
        combinedInput = concatenate([gate_1.output, net_2.output,gate_3.output,net_4.output,gate_5.output,net_6.output,\
                                    gate_7.output,net_8.output,gate_9.output,net_10.output])
        combinedInput = Reshape((10,8), input_shape=(8*10,))(combinedInput)
        lstm = LSTM(units=16, dropout=drop_out, recurrent_dropout=drop_out,return_sequences=False, kernel_initializer=init)
        output = lstm(combinedInput)
        output = Dense(1, activation='linear', kernel_initializer= init1)(output)
        model = Model(inputs=[gate_1.input, net_2.input,gate_3.input,net_4.input,gate_5.input,net_6.input\
                                    ,gate_7.input,net_8.input,gate_9.input,net_10.input], outputs=output)
        model.compile(loss='mean_squared_error',optimizer=optimizer)
        model.summary()
        train_history = model.fit(x=[i for i in x_train],y=y_train,epochs=epochs,validation_data=(x_valid, y_valid),batch_size=batch_size)
        val_loss = train_history.history['val_loss'][-1]
        score.append(val_loss)
    return(np.mean(score))