"""###########################################################################
############## Traffic State Prediction Module             ###################
############## defined  machine learning models            ###################
###########################################################################"""

import pandas as pd
import inspect
import os
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import control_panel as cp

#turnoff overwrite dataframe warning
pd.options.mode.chained_assignment = None  # default='warn'

class offline_model:
    
    #   initialize model
#    def __init__(self):

    #cong_state_type: 'cluster' or 'tti'
    #pred_horizon: 1-6 (10 - 60 second)
    def base_model_simple(self, cong_state_type, pred_horizon):
        
        module_path = inspect.getfile(inspect.currentframe())
        module_dir = os.path.realpath(os.path.dirname(module_path))
        
        if(cp.mac == 1):
            data = pd.read_csv("%s/IO/tm_df_base_lagged.csv" %module_dir)
        else:
            data = pd.read_csv("%s\\IO\\tm_df_base_lagged.csv" %module_dir)
        
        #   creatng train, test data sets
        train_x = data[['mean_speed_tlag%d_slead1' %pred_horizon,
                         'ssd_tlag%d_slead1' %pred_horizon,
                         'mean_speed_tlag%d' %pred_horizon,
                         'mean_headway_tlag%d' %pred_horizon]]
       
        if(cong_state_type == 'cluster'):
            train_y = data['traffic_state_cluster']
        elif(cong_state_type == 'tti'):
            train_y = data['traffic_state_tti']
     
        rf_model = RandomForestClassifier(n_estimators = 500, max_depth=2, random_state=0)
        rf_model.fit(train_x, train_y)
        
        return rf_model
    
def optimal_spd_pred():
    module_path = inspect.getfile(inspect.currentframe())
    module_dir = os.path.realpath(os.path.dirname(module_path))
    
    if(cp.mac == 1):
        data = pd.read_csv("%s/IO/spdhrm_df_base.csv" %module_dir)
    else:
        data = pd.read_csv("%s\\IO\\spdhrm_df_base.csv" %module_dir)
    
    #   creatng train, test data sets
    train_x = data[['speed_congestion','speed_500', 'speed_1000', 'speed_1000']]
    train_y = data['VSL']
 
    rf_model = RandomForestClassifier(n_estimators = 500, max_depth=9, random_state=0)
    rf_model.fit(train_x, train_y)
    return rf_model