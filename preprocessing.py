from sklearn import preprocessing
import math
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
class preprocessing_data():
    """
    This class is the feature engineering part and it delete columns if they have the same 
    value, scaling the numerical values,transform the columns if they have only two values 
    to categorical and transform the locational data into clusters.
    """

    def __init__(self,dataframe,train_split):
        self.init_data = dataframe
        self.X_columns = ['cell_threshold_index', 'cell_type','cell_min_rise_delay', 'number_of_pins_gate', 'area_gate', 'cell_min_fall_delay',
       'cell_max_fall_delay', 'cell_max_rise_delay', 'height', 'width',
       'temperature_max', 'y_location', 'cell_strength_index', 'cell_voltage',
       'x_location', 'temperature_min', 'ID', 'cell_name',
       'net_resistance_max', 'pin_capacitance_max', 'pin_capacitance_max_fall',
       'ba_capacitance_max', 'M5', 'M4', 'M7', 'pin_capacitance_max_rise',
       'M1', 'M3', 'M2', 'total_capacitance_max', 'number_of_pins_net', 'area_net',
       'number_of_leaf_loads', 'pin_capacitance_min', 'ba_resistance_min',
       'total_capacitance_min', 'pin_capacitance_min_rise',
       'wire_capacitance_max', 'wire_capacitance_min', 'ba_resistance_max',
       'ID', 'base_name', 'number_of_wires', 'pin_capacitance_min_fall','M6','net_name',
       'number_of_leaf_drivers', 'net_resistance_min', 'ba_capacitance_min']
        

    def filter_columns(self,X):
        '''
        Return X: the dataframe that deleted the column that has same values

        '''
        X.columns = self.X_columns
        X = X.drop(columns=['ID','temperature_max','temperature_min','cell_name','M1','M5','base_name','number_of_leaf_drivers',\
                  'M6', 'net_name','cell_threshold_index','cell_type'])
        X = X.drop(columns=['number_of_pins_gate','height'])
        return X
        
    def normalise(self,X):
        '''
        Return X: the dataframe that has numerical columns scaled using:
        X_std * (X_max - X_min) + X_min

        '''
        min_max_scaler = preprocessing.MinMaxScaler()
        cell_min_rise_delay = X['cell_min_rise_delay'].values.reshape((-1, 1))
        cell_min_rise_delay_scaled = min_max_scaler.fit_transform(cell_min_rise_delay)
        X['cell_min_rise_delay'] = cell_min_rise_delay_scaled
        X[['cell_min_fall_delay','cell_max_fall_delay','cell_max_rise_delay']] = min_max_scaler.fit_transform(X[['cell_min_fall_delay','cell_max_fall_delay','cell_max_rise_delay']])
        X[['net_resistance_max','pin_capacitance_max','pin_capacitance_max_fall','ba_capacitance_max']] = min_max_scaler.fit_transform(X[['net_resistance_max','pin_capacitance_max','pin_capacitance_max_fall','ba_capacitance_max']])
        X[['M2','M3','M4','M7']] = min_max_scaler.fit_transform(X[['M2','M3','M4','M7']])
        X[['pin_capacitance_max_rise','total_capacitance_max','number_of_pins_net','area_net','number_of_leaf_loads','pin_capacitance_min',\
         'ba_resistance_min','total_capacitance_min','pin_capacitance_min_rise','wire_capacitance_max','wire_capacitance_min','ba_resistance_max',\
          'number_of_wires','pin_capacitance_min_fall','net_resistance_min','ba_capacitance_min']] \
        = min_max_scaler.fit_transform(X[['pin_capacitance_max_rise','total_capacitance_max','number_of_pins_net','area_net','number_of_leaf_loads','pin_capacitance_min',\
         'ba_resistance_min','total_capacitance_min','pin_capacitance_min_rise','wire_capacitance_max','wire_capacitance_min','ba_resistance_max',\
          'number_of_wires','pin_capacitance_min_fall','net_resistance_min','ba_capacitance_min']])
        return X
    
    
    def categorical(self,X):
        '''
        Return X: the dataframe that has the columns transformed to 0/1 variable if they
        only have two values

        '''
        M4 = X['M4'].values
        M7 = X['M7'].values
        M3 = X['M3'].values
        M2 = X['M2'].values
        area_gate = X['area_gate'].values
        width = X['width'].values
        voltage = X['cell_voltage'].values
        strength = X['cell_strength_index'].values
        area_gate_new = [1 if i==3.812160 else 0 for i in area_gate]
        X['area_gate'] = area_gate_new 
        width_new = [1 if i==50136 else 0 for i in width]
        X['width'] = width_new 
        strength_new = [1 if i==39744 else 0 for i in strength]
        X['cell_strength_index'] = strength_new 
        voltage_new = [1 if i==0.95 else 0 for i in voltage]
        X['cell_voltage'] = voltage_new
        X['M4'] = [0 if math.isnan(i) else i for i in M4]
        X['M7'] = [0 if math.isnan(i) else i for i in M7]
        X['M3'] = [0 if math.isnan(i)else i for i in M3]
        X['M2'] = [0 if math.isnan(i) else i for i in M2]
        return X
    
    def location_transform(self,X):
        '''
        Return X: the datafram that has transformed the locational data using kmeans algorithm
        
        '''
        x_location = X['x_location'].values
        y_location = X['y_location'].values
        X['x_location'] = [0 if i=='pin' else i for i in x_location]
        X['y_location'] = [0 if i=='pin' else i for i in y_location]
        coords = list(zip(X['x_location'].values,X['y_location'].values))
        kmeans = MiniBatchKMeans(n_clusters=10,random_state=0,batch_size=6,max_iter=10).fit(coords)
        new_coord = kmeans.predict(coords)
        X['location'] = new_coord
        X = X.drop(columns=['x_location','y_location'])
        new_loc = pd.get_dummies(X['location'], prefix='location').drop(columns=['location_0'])
        X = pd.concat([X,new_loc],axis=1)
        X = X.drop(columns=['location'])
        return X
        