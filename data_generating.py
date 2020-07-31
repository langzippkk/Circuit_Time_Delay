import json
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
import math
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class data_generating:
    def __init__(self,jsonfile):
        self.jsonfile = jsonfile
        self.json_data = self.get_Data()
        self.gate_attribute,self.gate_attribute2 = self.get_Gate()
        self.net_attribute = self.get_Net()
        self.gate_len = len(self.gate_attribute)
        self.net_len = len(self.net_attribute)
#         self.X,self.Y = self.generate_template_X()
        self.name = self.gate_attribute+self.net_attribute
        self.X,self.Y = self.generate_short_X()

    def get_Data(self):
        '''
        Decoding the json file one by one
        Return json_data: decoded data
        '''
        json_data = []
        file = open(self.jsonfile)
        for line in file:
            json_line = json.loads(line)
            json_data.append(json_line)
        return json_data


    def get_Gate(self):
        '''
        Return attribute_gate: feature name for sequence1 and 2, 
        for building the model, return two different name lists for the same features.
        '''
        attribute_gate = []
        attribute_gate2 = []
        for i in (((self.json_data[0])['input']['Sequence 1'])[0][0]).items():
            attribute_gate.append(i[0])
            attribute_gate2.append(i[0]+'_2')
        return attribute_gate,attribute_gate2
    

    def get_Net(self):
        '''
        Return attribute_net: list of net feature names
        '''
        attribute_net = []
        for i in (((self.json_data[0])['input']['Sequence 1'])[0][1]).items():
            attribute_net.append(i[0])
        #print(attribute_net)
        return attribute_net
    
    def generate_template_X(self):
        '''
        Transform the data from json to tensors
        Return X:
        (5196 data* 3 sequence/data*sequence length)
        Return Y:
        (5196 data *3 sequence)
        
        '''
        X = []
        Y = []
        for three_seq in self.json_data:
            ## 3 sequences
            temp1 = three_seq['input']['Sequence 1']
            temp2 = three_seq['input']['Sequence 2']
            temp3 = three_seq['input']['Sequence 3']
            
            ## get the input
            X_1,X_2,X_3 = ([] for i in range(3)) 
            for i in range(len(temp1[0])):
                X_1.append(list(temp1[0][i].values()))
            for j in range(len(temp2[0])):
                X_2.append(list(temp2[0][j].values()))
            for k in range(len(temp3[0])):
                temp4 = temp3[0][k]
                if 'rise_fall' in temp4:
                    del temp4['rise_fall']
                X_3.append(list(temp4.values()))
            Y_1 = (float(temp1[1]['value']))
            Y_2 = (float(temp2[1]['value']))
            Y_3 = (float(temp3[1]['value']))
            X.append([X_1,X_2,X_3])
            seq_len = len(X_1)+len(X_2)+len(X_3)
            Y.append([Y_1,Y_2,Y_3,len(X_1),len(X_2),len(X_3)])
        return X,Y
    
    
    def generate_alternating_df(self,seq_len):
        '''Generating dataframe for LSTM
        Initiate a dataframe with columns name of gate and net
        Return initial_df: dataframe with one get and one net data in each row
        '''
        initial_df = pd.DataFrame([0]*(len(self.gate_attribute)+len(self.net_attribute))).transpose()
        initial_df.columns = self.name
        for ex in range(len(self.X)):
            if (len(initial_df)%1000==1):
                print(len(initial_df))
            for seq in range(seq_len):
                temp = self.X[ex][seq]
                for i in range(0,len(temp)-1,2):
                    final = (pd.DataFrame(temp[i]+temp[i+1]).transpose())
                    final.columns = self.name
                    initial_df = pd.concat([initial_df,final])
        return initial_df
        
        
        
    def generate_short_X(self):
        '''
        Similar to generate_alternating_df(), but only get the data from sequence1 and 2.
        Return X:
        (5196 data, 2 sequence/data,sequence length)
        Return Y:
        (5196 data *2 sequence)
        
        '''
        X = []
        Y = []
        for three_seq in self.json_data:
            ## 2 sequences
            temp1 = three_seq['input']['Sequence 1']
            temp2 = three_seq['input']['Sequence 2']
            ## get the input
            X_1,X_2,X_3 = ([] for i in range(3)) 
            if (len(temp1[0])==10):
                for i in range(len(temp1[0])):
                    X_1.append(list(temp1[0][i].values()))
            if (len(temp2[0])==10):
                for j in range(len(temp2[0])):
                    X_2.append(list(temp2[0][j].values()))
            Y_1 = (float(temp1[1]['value']))
            Y_2 = (float(temp2[1]['value']))
            X.append([X_1,X_2])
            Y.append([Y_1,Y_2,len(X_1),len(X_2)])
            
        return X,Y


    def generate_sequence3(self):
        '''
        Similar to generate_alternating_df(), but only get the data from sequence3.
        Return two X:
        (5196 data, 1 sequence/data,sequence length)*2
        Return Y:
        (5196 data *1 sequence)
        
        '''
        X = []
        Y = []
        for three_seq in self.json_data:
            ## 2 sequences
            temp1 = three_seq['input']['Sequence 3']
            ## get the input
            X_3 = []
            for i in range(len(temp1[0])):
                X_3.append(list(temp1[0][i].values()))
            Y_3 = (float(temp1[1]['value']))
            X.append(X_3)
            Y.append([Y_3,len(X_3)])
            
        return X,Y