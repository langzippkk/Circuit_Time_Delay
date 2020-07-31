import numpy as np
from preprocessing import preprocessing_data
from data_generating import data_generating
import pandas as pd

def test_train(X,Y):
    '''
    Input: X and Y dataframe after using data_generating class.
    X shape: (number of gate-net,49 unprocessed features)
    Y shape: (number of labels,)
    Split the data into train and test with split ratio 80/20 and
    using the preprocessing_data class to transform the data for each of test and train data.
    Return X_train,Y_train,X_test,Y_test: The splitted train/test tensors
    X_train shape (number of gate-net*0.8,41 processed features))
    Y_train shape (number of labels*0.8,)
    X_test shape  (number of gate-net*0.2,41 processed features))
    Y_test shape (number of labels*0.2,)

    '''
    split_Y = int(len(Y)*0.8)
    split = split_Y*10
    X_train = X[:split]
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
    X_train = generating_tensor_X(X_train)
    Y_train = generating_tensor_Y(Y_train)
    X_test = generating_tensor_X_test(X_test)
    Y_test = generating_tensor_Y(Y_test)
    return X_train,Y_train,X_test,Y_test


def generating_tensor_X_SVM(X):
    '''
    Input: The X dataframe with 1 gate and 1 net each row.
    X shape: (number of gate-net,41 processed features)
    Return batch: shape(number of sequences,(24+17)*5 features),
    the tensor that using 5 gates and 5 nets in a row

    '''
    batch = []
    for i in range(1,len(X)-3,5):
        temp = np.concatenate((X.loc[i].values,X.loc[i+1].values,X.loc[i+2].values,\
                              X.loc[i+3].values,X.loc[i+4].values))
        batch.append(temp)
    return batch
    
def generating_tensor_X(X):
    '''
    Input: The X dataframe with 1 gate and 1 net each row.
    X shape (number of gate-net,41 features)
    Return batch for LSTM model: shape(number of sequences,4 sliding window,(24+17)*2 features),
    the tensor that using 2 gates and 2 nets in a single sliding windows

    '''
    batch = []
    for i in range(1,len(X)-3,5):
        start = i
        time_batch = []
        for j in range(4):
            temp = np.concatenate((X.loc[start+j].values,X.loc[start+j+1].values))
            time_batch.append(temp)
        batch.append(time_batch)
    return batch

def generating_tensor_Y(Y):
    '''
    Input: The Y dataframe with 2 labels in a row.
    Y shape: (number of gate-net,2)
    Return label : shape(number of sequences,1)

    '''
    label = []
    for i in range(len(Y)):
        temp = Y.loc[i].values
        label.append(temp[0])
        label.append(temp[1])
    return label

def generating_CNN_intput(X):
    '''
    Input:generating_tensor_X's output with shape (number of sequences,4 sliding window, (24+17)*2 features)
    Output:Reshaped output for each sliding window, with shape(number of sequences,4,2,41)
    '''
    temp1,temp2,temp3,temp4 =  ([] for i in range(4)) 
    for i in range(len(X)):
        temp1.append(X[i][0].reshape(2,41))
        temp2.append(X[i][1].reshape(2,41))
        temp3.append(X[i][2].reshape(2,41))
        temp4.append(X[i][3].reshape(2,41))
    return ([np.array(temp1),np.array(temp2),np.array(temp3),np.array(temp4)])

def test_train_Multi(X,Y):
    '''    
    Input: X and Y dataframe after using data_generating class.
    X shape: (number of gate-net,49 unprocessed features)
    Y shape: (number of labels,)
    Split the data into train and test with split ratio 80/20 and
    using the preprocessing_data class to transform the data for each of test and train.
    And then, split gate and net data into seperate dataframes
    Return dataframes: gate_train,gate_test,net_train,net_test,Y_train,Y_test
    gate_train shape: (number of gate-net*0.8,17 processed features)
    gate_test shape:(number of gate-net*0.2,17 processed features)
    net_train shape:(number of gate-net*0.8,24 processed features)
    net_test shape:(number of gate-net*0.2,24 processed features)

    '''
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
    Y_train = generating_tensor_Y(Y_train)
    Y_test = generating_tensor_Y(Y_test)
    gate_train = pd.concat([X_train.iloc[:,:8],X_train.iloc[:,-9:]],axis=1)
    net_train = X_train.iloc[:,8:-9]
    gate_test =  pd.concat([X_test.iloc[:,:8],X_test.iloc[:,-9:]],axis=1)
    net_test = X_test.iloc[:,8:-9]
    return gate_train,gate_test,net_train,net_test,Y_train,Y_test


def generating_tensor_CNN(X):
    '''
    Input: Seperated gate and net dataframes
    gate_train shape: (number of gate-net*0.8,17 processed features)
    gate_test shape:(number of gate-net*0.2,17 processed features)
    net_train shape:(number of gate-net*0.8,24 processed features)
    net_test shape:(number of gate-net*0.2,24 processed features)

    Transform the input dataframe to tensors
    Return res: tensors of shape (number of sequences,length of gate/net features)

    '''
    res = []
    for i in range(1,len(X)-3,5):
        batch = []
        batch.append([X.loc[i].values,X.loc[i+1].values,X.loc[i+2].values,X.loc[i+3].values,X.loc[i+4].values])
        res.append(batch)
    return res


def generate_combined(gate_train,net_train):
    '''
    Input: gate and net data
    gate_train shape: (number of sequences,length of gate features)
    net_train shape: (number of sequences,length of net features) 
    Extract each gate and net and reshape it for Dense net
    Return list: 10 concatenated tensors with shape (number of sequences,10,1,17/24)

    '''
    input1_train = np.asarray([i[0][0].reshape(1,17) for i in gate_train])
    input2_train = np.asarray([i[0][0].reshape(1,24) for i in net_train])
    input3_train = np.asarray([i[0][1].reshape(1,17)for i in gate_train])
    input4_train = np.asarray([i[0][1].reshape(1,24) for i in net_train])
    input5_train = np.asarray([i[0][2].reshape(1,17) for i in gate_train])
    input6_train = np.asarray([i[0][2].reshape(1,24) for i in net_train])
    input7_train = np.asarray([i[0][3].reshape(1,17) for i in gate_train])
    input8_train = np.asarray([i[0][3].reshape(1,24) for i in net_train])
    input9_train = np.asarray([i[0][4].reshape(1,17) for i in gate_train])
    input10_train = np.asarray([i[0][4].reshape(1,24) for i in net_train])
    return ([input1_train,input2_train,input3_train,input4_train,input5_train,input6_train,input7_train,input8_train,input9_train,input10_train])