#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tuning_case_study.py

Assesses the impact on performance by adjusting the decision threshold 
probability (DTP)

Created on Thu Aug 11 15:44:40 2022

@author: rtsearcy
"""

import random
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
import joblib

from beach_model import model_eval
from model_tuner import model_tuner

# Functions
def caps_off(axx): ## Turn off caps on boxplots
    lines = axx.lines
    for i in range(0, int(len(lines)/6)):
        lines[(i*6)+2].set_color('none')
        lines[(i*6)+3].set_color('none')

def flier_shape(axx, shape='.'):  ## Set flier shape on boxplots
    lines = axx.lines
    for i in range(0, int(len(lines)/6)):
        lines[(i*6)+5].set_marker(shape)

def plot_spines(axx, offset=8): # Offset position, Hide the right and top spines
    axx.spines['left'].set_position(('outward', offset))
    axx.spines['bottom'].set_position(('outward', offset))
    
    axx.spines['right'].set_visible(False)
    axx.spines['top'].set_visible(False)


### Input Cases to Evaluate
folder = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting'

n_models = 20  # number of models (per model type) to assess
model_type = ['lm','svm','rf','gbm']

seed = 0   # random seed

maximize = 'sens'  # tune to maximize sens or spec
min_spec = 0.8     # used if maximize = sens
min_sens = 0.3     # used if maximize = spec


### Select models randomly from combinations

# load performance file from untuned models to select cases from
perf_file_name = os.path.join(folder, 'tables_figs','performance_metrics_all.csv')
df_perf = pd.read_csv(perf_file_name)
df_perf['idx'] = df_perf.index

df_before = pd.DataFrame()
df_after = pd.DataFrame()

for i in range(0, len(model_type)):
    m = model_type[i]
    print(m.upper())
    
    df_choose = df_perf[(df_perf.model == m) & 
                        (df_perf.lead_time > 0) &
                        (df_perf.dataset == 'test')]  # exclude nowcast
    df_choose.reset_index(inplace=True)
    
    random.seed(i + seed)
    idx = random.sample(range(0, len(df_choose)), n_models)
    # print('random indices: ')
    # print(idx)
    
    for j in idx:
        df = df_choose.iloc[j]      

### Load models
        par_folder = os.path.join(folder,
                                  'beach',
                                  df.beach,
                                  'models',
                                  df.FIB,
                                  df.partition
                                  )
        train_data = pd.read_csv(os.path.join(par_folder, 'train_data.csv'), index_col=['date']) 
        test_data = pd.read_csv(os.path.join(par_folder, 'test_data.csv'), index_col=['date'])
        
        model_folder = os.path.join(par_folder, 
                                    'lt_' + str(df.lead_time),
                                    df.model)
        
### Performance before tuning
        bef_perf = pd.read_csv(os.path.join(model_folder, m + '_performance.csv'), index_col=[0])
        bef_perf.index.name = 'dataset'
        bef_perf.reset_index(inplace=True)
        
        meta = {'beach':df.beach, 
                'FIB':df.FIB, 
                'partition': df.partition,
                 'lead_time': int(df.lead_time), 
                 'model': m}
        
        bef_perf = bef_perf.join(pd.DataFrame(data=meta, 
                                                   index=bef_perf.index), 
                                                   how = 'left')
        df_before = df_before.append(bef_perf)  # Add to aggregated df
        
        print('\n')
        print(meta)
        print(bef_perf[['sens','spec','AUC','N','exc']])
        
### Tuning
        ## Features
        temp_features = pd.read_csv(os.path.join(model_folder, 
                                                  m + '_features.csv'))
        env_vars = list(temp_features.features)

        model = joblib.load(os.path.join(model_folder, m + '_model.pkl')) # load model pkl
        
        ## Scale Vars
        scaler = StandardScaler()
        scaler.fit(train_data)
        X_train = pd.DataFrame(data=scaler.transform(train_data), index=train_data.index, columns=train_data.columns)
        X_train = X_train[env_vars]
        X_test = pd.DataFrame(data=scaler.transform(test_data), index=test_data.index, columns=test_data.columns)
        X_test = X_test[env_vars]
        
        ## Run Tuner
        DTP = model_tuner(model, X_train, train_data[df.FIB +'_exc'], 
                          maximize=maximize, min_sens=min_sens, min_spec=min_spec, scale=False)
        train_pred = model.predict_proba(X_train[env_vars])[:,1]
        test_pred = model.predict_proba(X_test[env_vars])[:,1]
        
        print('\nDTP: ' + str(DTP))

### Tuned performance
        
        ## Test at new decision threshold
        train_tune = model_eval(train_data[df.FIB + '_exc'], train_pred, thresh=DTP)
        test_tune = model_eval(test_data[df.FIB + '_exc'], test_pred, thresh=DTP)
        
        ## Add to After DF
        tune_perf = pd.DataFrame(data=[train_tune, test_tune], index=['train','test'])
        tune_perf.index.name = 'dataset'
        tune_perf.reset_index(inplace=True)
        
        # Add metadata
        tune_perf = tune_perf.join(pd.DataFrame(data=meta, 
                                                  index=tune_perf.index), 
                                     how = 'left')
        tune_perf['DTP'] = DTP
        df_after = df_after.append(tune_perf)  # Add to aggregated df

        print(tune_perf[['sens','spec','AUC','N','exc']])


#%% Print Results

df_before.reset_index(inplace=True)
df_after.reset_index(inplace=True)

print('\nOverall:')
print('\nBEFORE')
print(df_before[df_before.dataset=='test'].describe()[['sens','spec']].round(3))
print('\nAFTER')
print(df_after[df_after.dataset=='test'].describe()[['sens','spec']].round(3))

print('\nBy model:')
print('\nBEFORE')
print(df_before[df_before.dataset=='test'].groupby('model').describe()[['sens','spec','AUC']].loc[:, pd.IndexSlice[:,['50%','25%', '75%']]].round(2))
print('\nAFTER')
print(df_after[df_after.dataset=='test'].groupby('model').describe()[['sens','spec','AUC']].loc[:, pd.IndexSlice[:,['50%','25%', '75%']]].round(2))

## DTP
df_after[df_after.dataset=='test'].DTP.describe()

df_before.set_index(['beach','FIB','partition','dataset','lead_time','model'], inplace=True)
df_after.set_index(['beach','FIB','partition','dataset','lead_time','model'], inplace=True)

df_change = df_after - df_before
df_change['DTP'] = df_after.DTP
df_change.describe().round(3)[['sens','spec','DTP']]

# %% Plot Results

### Change in metrics by model
sns.boxplot(x='model', y = 'sens', data = df_change.reset_index())

sns.scatterplot(x='spec',y='sens', hue='model',size='DTP', data=df_change.reset_index())

