#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregates performance metrics for forecast models and computes
statistics

Created on Fri Jul  1 11:27:32 2022

@author: rtsearcy
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
import joblib
from beach_model import model_eval

### Input Cases to Evaluate
base_folder = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting'

beach = ['Cowell', 'HSB3N']#,'Manhattan']
fib = ['EC','ENT']#,'ENT']
model_types = ['per','lm','svm', 'rf', 'gbm']

tune=True
tune_prob = 0.25  # prediction threshold used for testing how tuned models would perform

### Plot parameters / functions
if beach == 'Cowell':
    beach_name = 'Cowell Beach'
    abbrev = 'CB'
elif beach == 'HSB3N':
    beach_name = 'Huntington State Beach'
    abbrev = 'HSB'

params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 10.5,
   'xtick.labelsize': 10,
   'ytick.labelsize': 11,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

## Colors

# For models
model_colors = {
    'per': '#003f5c',
    'lm': '#565089',
    'svm': '#b1568f', 
    'rf': '#f3696c',
    'gbm': '#fda237'
    }

# model_colors = {
#     'per': '#8dd3c7',
#     'lm': '#ffffb3',
#     'svm': '#bebada', 
#     'rf': '#fb8072',
#     'gbm': '#80b1d3'
#     }

colors_all = [model_colors[c] for c in model_colors]
colors_perf_compare = [model_colors[c] for c in ['lm','svm','rf','gbm']]

pal_grey = ['#969696','#525252']  # grey, black
pal = ['#de425b','#2c8380']
pal4c = ['#253494','#2c7fb8','#41b6c4','#a1dab4'] # 4 color blue tone
#pal = sns.color_palette(pal)

### Functions
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


# %% Aggregate Results
perf_file_name = os.path.join(base_folder, 'tables_figs','performance_metrics_all.csv')
tune_file_name = os.path.join(base_folder, 'tables_figs','performance_metrics_all_tuned_' + str(int(100*tune_prob)) + '.csv')
feature_file_name = os.path.join(base_folder, 'tables_figs','model_features_all.csv')

try:
    df_perf = pd.read_csv(perf_file_name)
    df_features = pd.read_csv(feature_file_name)
    if tune:
        df_tune = pd.read_csv(tune_file_name)
    
except:
    df_perf = pd.DataFrame()
    df_tune = pd.DataFrame()
    df_features = pd.DataFrame()
    
    ### Iterate
    for b in beach:
        print('\n' + b)
        model_base_folder = os.path.join(base_folder, 'beach', b, 'models')
        
        for f in fib:
            print('\n' + f)
            fib_folder = os.path.join(model_base_folder, f)
            if not os.path.isdir(fib_folder):  # if folder doesn't exist
                continue
            partitions = [s for s in os.listdir(fib_folder) if 'train' in s]
            
            for p in partitions:
                print(p)
                par_folder = os.path.join(fib_folder, p)
                lead_times = [lt for lt in os.listdir(par_folder) if 'lt' in lt]
                if tune:
                    train_data = pd.read_csv(os.path.join(par_folder, 'train_data.csv'), index_col=['date']) 
                    test_data = pd.read_csv(os.path.join(par_folder, 'test_data.csv'), index_col=['date']) 
                
                for lt in lead_times:
                    print(' LT - ' + str(lt))
                    models = [m for m in os.listdir(os.path.join(par_folder, lt)) if m in model_types]
                    if len(models) == len(model_types):
                        models = model_types
                    
                    for m in models:
                        model_folder = os.path.join(par_folder, lt, m)
                        
                        ## Model metadata
                        meta = {'beach':b, 'FIB':f, 'partition': p,
                                'lead_time': int(lt[-1]), 'model': m}
                        
                        ### Performance
                        model_perf = pd.read_csv(os.path.join(model_folder, m + '_performance.csv'), index_col=[0])
                        model_perf.index.name = 'dataset'
                        model_perf.reset_index(inplace=True)
                        
                        # Add metadata
                        model_perf = model_perf.join(pd.DataFrame(data=meta, 
                                                                  index=model_perf.index), 
                                                     how = 'left')
                        df_perf = df_perf.append(model_perf)  # Add to aggregated df
                        
                        if m != 'per':
                            
                            ### Features
                            temp_features = pd.read_csv(os.path.join(model_folder, 
                                                                      m + '_features.csv'))
                            model_features = meta.copy()
                            env_vars = list(temp_features.features)
                            model_features['feature'] = env_vars
                            
                            df_features = df_features.append(pd.DataFrame(model_features))
                            
                            if tune: ### Tune
                                bin_class = joblib.load(os.path.join(model_folder, m + '_model.pkl')) # load model pkl
                                #env_vars = list(bin_class.feature_names_in_) # get variables
                                
                                ## Scale Vars
                                scaler = StandardScaler()
                                scaler.fit(train_data)
                                X_train = pd.DataFrame(data=scaler.transform(train_data), index=train_data.index, columns=train_data.columns)
                                X_train = X_train[env_vars]
                                X_test = pd.DataFrame(data=scaler.transform(test_data), index=test_data.index, columns=test_data.columns)
                                X_test = X_test[env_vars]
                                
                                ## probability predictions
                                train_pred = bin_class.predict_proba(X_train[env_vars])[:,1]
                                test_pred = bin_class.predict_proba(X_test[env_vars])[:,1]
                                
                                ## Test at new decision threshold
                                train_tune = model_eval(train_data[f+'_exc'],train_pred, thresh=tune_prob)
                                test_tune = model_eval(test_data[f+'_exc'],test_pred, thresh=tune_prob)
                                
                                ## Add to Tune DF
                                tune_perf = pd.DataFrame(data=[train_tune,test_tune], index=['train','test'])
                                tune_perf.index.name = 'dataset'
                                tune_perf.reset_index(inplace=True)
                                # Add metadata
                                tune_perf = tune_perf.join(pd.DataFrame(data=meta, 
                                                                          index=tune_perf.index), 
                                                             how = 'left')
                                df_tune = df_tune.append(tune_perf)  # Add to aggregated df
                                
                        else: # add PER perf to tune df, too
                            if tune:
                                df_tune = df_tune.append(model_perf)
    
    ### Organize Data
    meta_cols = list(meta.keys()) + ['dataset']
    perf_metrics = [c for c in df_perf if c not in meta_cols]
    df_perf = df_perf[meta_cols + perf_metrics].reset_index(drop=True)
    if tune:
        df_tune = df_tune[meta_cols + perf_metrics].reset_index(drop=True)
    
    ### Save
    df_perf.to_csv(perf_file_name, index=False)
    df_tune.to_csv(tune_file_name, index=False)
    df_features.to_csv(feature_file_name, index=False)

## Convert model acronyms
model_map = {
    'lm': 'BLR',
    'svm': 'SVM',
    'rf': 'RF',
    'gbm': 'GBM',
    'per': 'PER'
    }
df_perf['model'] = df_perf['model'].map(model_map)
if tune:
    df_tune['model'] = df_tune['model'].map(model_map)
df_features['model'] = df_features['model'].map(model_map)

#%% Data Partitions
partitions = df_perf.groupby(['beach','FIB','partition','dataset']).first()[['N','exc']]
partitions.reset_index(inplace=True)

# manually make table
partitions.pivot(index='partition', 
                 columns = ['beach','FIB','dataset'], 
                 values=['N','exc']).to_csv(os.path.join(base_folder, 'tables_figs', 'partitions_raw.csv'))

### Plot partitions breakdown
partitions['%'] = (100*partitions.exc / partitions.N).round(1)
#partitions['partition_idx'] = [c[6:8] + '-' + c[9:11] + '/' + c[17:19] + '-' + c[20:22] for c in partitions.partition]
partitions['partition_idx'] = partitions.partition.map(dict(zip(partitions.partition.unique(), range(1,len(partitions.partition.unique())+1))))

sns.catplot(x='partition_idx',y='%', data=partitions,
            hue='dataset', row='beach', col='FIB',
            hue_order=['train','test'], row_order=beach, col_order=fib,
            kind='bar', palette=pal_grey, saturation=1, height=2, aspect=2.5)

plt.tight_layout()
#sns.barplot(x='partition',y='N',hue='dataset', data=partitions)

#%% Number of Models
n_for_models = len(df_perf[(df_perf.lead_time.isin([1,2,3])) & (df_perf.model != 'per') & (df_perf.dataset != 'test')])

n_total_models = len(df_perf[(df_perf.model != 'per') & (df_perf.dataset != 'test')])

# %% Summarize Features
'''
Some questions to focus research:
- Which environmental parameters are most important in forecasting FIB?
-   Or: how do the relationships between FIB and EVs evolve in time?
 
'''

### Load Feature Metadata
feat_met = pd.read_csv(os.path.join(base_folder,'tables_figs','feature_metadata.csv'))
len(feat_met) # 347 unique features
feat_met.type.unique() # 7 feature types (not counting FIB and upwell)

df_feat = df_features.merge(feat_met, how = 'left', on='feature')

df_feat = df_feat[df_feat.lead_time != 0] # Remove nowcast

n_part = len(df_feat.partition.unique())
n_model = len(df_feat.model.unique()) 
n_lead =  len(df_feat.lead_time.unique())
n = n_part * n_lead * n_model

### Analyze Number  model features
len(df_feat.feature.unique()) # 130 (161): Num unique features in forecast (all) models

# Avg num features per model
avg_features = df_feat.groupby(['beach','FIB','partition','lead_time','model']).count()['feature'].reset_index()

avg_features.groupby('model').describe().round(0)['feature'] # Mean features by model type 
avg_features.mean()['feature'].round(0) # (=10 across all models)
sns.boxplot(x='model', y='feature', hue='lead_time', data=avg_features)
plt.tight_layout()


### Top 10 features
X = df_feat.groupby('feature').count()['FIB'].sort_values(ascending=False).head(10) # All models

X = df_feat.groupby(['beach','FIB','feature']).count()['model'].sort_values(ascending=False)


X.xs('Cowell').xs('EC').head(10)
X.xs('Cowell').xs('ENT').head(10)
X.xs('HSB3N').xs('EC').head(10)
X.xs('HSB3N').xs('ENT').head(10)


### Most common feature type in models
feat_type = df_feat.groupby(['beach','FIB','partition','lead_time','model','type']).count()['feature'].reset_index()

#feat_type.groupby('type').sum()['feature'] / len(df_features)  # % of total features in each category

# % of models with feature of each type
feat_type.groupby('type').count()['feature'] / len(df_feat.groupby(['beach','FIB','partition','lead_time','model']).count().index)

sns.barplot(x='type',y='feature',hue='model', data = (feat_type.groupby(['type','model']).count() / (n)).reset_index())  # across all beaches and FIB

sns.catplot(x='type',y='feature', col ='lead_time', data=feat_type, kind='bar')

sns.catplot(x='type',y='feature',hue='model',row='beach',col='FIB', kind='bar',
            data=(feat_type.groupby(['beach','FIB','type','model']).count() / (n_lead*n_part)).reset_index())

sns.catplot(x='type',y='feature',row='beach',col='FIB', kind='bar',
            data=(feat_type.groupby(['beach','FIB','type']).count() / (n)).reset_index())
plt.tight_layout()

### Lags vs lead time

# How affected by lead time


# %% Summarize Performance
'''
Some questions to focus research:
- Best model types 
- How does this change with lead time?

- Do forecast models have more skill/accuracy than the persistence model?
    - Which model architectures perform best?
- How do accuracy metrics change with increasing lead time (compared to nowcast)

- Does know recent FIB help improve accuracy?
- Optimal training set length? 2y,  3y, 5y, 10y
 
'''

index_cols = ['beach','FIB','partition','lead_time','model']
mets = ['sens','spec','AUC']

df_perf = df_perf[df_perf.dataset == 'test'] # Remove training data
df_for = df_perf[df_perf.lead_time != 0]


### Overall Performance
df_perf.groupby(['beach','FIB','partition','dataset']).std()[['N','exc']] # see that equivalent data between lead times
df_perf.groupby(['model','lead_time']).mean().round(2) # avg perf by model

# all models together
df_for[df_for.model != 'PER'].describe()[mets].round(2)

# By lead Times
df_for[df_for.model != 'PER'].groupby(['lead_time']).describe()[mets].loc[:, pd.IndexSlice[:,['50%','25%', '75%']]].round(2)
#df_for[df_for.model != 'PER'].groupby(['lead_time']).describe()[mets].loc[:, pd.IndexSlice[:,['mean','max', 'std']]].round(2)


## Performance by Model (across all beaches, FIB, partitions, and lead times)
df_for.groupby(['model']).describe()['sens'].loc[model_map.values()].round(2)
df_for.groupby(['model']).describe()['spec'].loc[model_map.values()].round(2)
df_for.groupby(['model']).describe()['AUC'].loc[model_map.values()].round(2)

df_for.groupby(['model']).describe()[mets].loc[:, pd.IndexSlice[:,['50%','25%', '75%']]].round(2)


## By beach, FIB, 
df_for.groupby(['beach','FIB']).describe()['AUC'].round(2)
df_for[df_for.model != 'PER'].groupby(['beach','FIB']).describe()[mets].loc[:, pd.IndexSlice[:,['mean','max']]].round(2)
df_for[df_for.model != 'PER'].groupby(['beach','FIB']).describe()[mets].loc[:, pd.IndexSlice[:,['50%','25%', '75%']]].round(2)
#compare to PER
df_for[df_for.model == 'PER'].groupby(['beach','FIB']).describe()[mets].loc[:, pd.IndexSlice[:,['50%','25%', '75%']]].round(2)


# and lead time
df_for.groupby(['beach','FIB','lead_time']).describe()['AUC'].round(2)
df_for[df_for.model != 'PER'].groupby(['beach','FIB','lead_time']).describe()[mets].loc[:, pd.IndexSlice[:,['mean','max']]].round(2)

### Boxplots of all performance metrics (by model, all beaches)
sns.catplot(x='lead_time',y='value', hue='model', col='variable', #row = 'FIB',
            data=df_for.melt(id_vars = ['lead_time','model','FIB'], value_vars = mets),
            kind='box', palette=colors_all, saturation=1, height=4, aspect=.75, 
            legend_out=True)

c=0
for a in plt.gcf().axes:
    plt.sca(a)
    
    plt.ylabel('')
    plt.xlabel('Lead Time')
    
    if mets[c] == 'sens':
        title = 'Sensitivity'
    elif mets[c] == 'spec':
        title = 'Specificity'
    elif mets[c] == 'bal_acc':
        title = 'Balanced Accuracy'
    else:
        title = 'AUC'
    plt.title(title)
    
    if mets[c] == 'AUC':
        plt.axhline(0.5, ls=':',c='k')
    
    caps_off(plt.gca())
    flier_shape(plt.gca())
    plot_spines(plt.gca())
    c+=1

plt.subplots_adjust(
    top=0.91,
    bottom=0.151,
    left=0.058,
    right=0.908,
    hspace=0.2,
    wspace=0.081)
#plt.tight_layout()


# Performance by model (across all lead times)
sns.catplot(x='model',y='value', col='variable', #row = 'lead_time',
            data=df_for.melt(id_vars = ['lead_time','model','FIB'], value_vars = mets),
            kind='box', palette=colors_all, saturation=1, height=4, aspect=.75, 
            legend_out=True)

### Frequency of top performance
cases = df_for.groupby(['beach','FIB','partition','lead_time']).count().index # cases (beach/FIB/partition/lead_time)
top_model = pd.DataFrame()
for c in cases:
    case = df_perf.loc[(df_perf.beach==c[0]) &
                       (df_perf.FIB == c[1]) &
                       (df_perf.partition == c[2]) &
                       (df_perf.lead_time == c[3])]
    
    temp = pd.DataFrame(index=[c], columns=mets)
    for m in mets:
        top = case.iloc[np.where(case[m] == case[m].max())]['model'].iloc[0]
        temp[m] = top
    top_model = top_model.append(temp)
top_model.index = cases
top_model.reset_index(inplace=True)

top_model.groupby('sens').count()['beach'].sort_values(ascending=False)
top_model.groupby('spec').count()['beach'].sort_values(ascending=False)
top_model.groupby('AUC').count()['beach'].sort_values(ascending=False)
    

### N models with AUC > 0.5
cols = ['model']
(df_for[df_for.AUC > 0.5].groupby(cols).count()['N'] / df_for.groupby(cols).count()['N']).round(3) # percentage

len(df_for[(df_for.AUC > 0.5) & (df_for.model != 'PER')]) / len(df_for[(df_for.model != 'PER')]) # overall

### Compared to Persistence
per_compare = pd.DataFrame()
per_perf = df_perf[df_perf.model == 'PER']
mod_perf = df_perf[df_perf.model != 'PER']


for p in per_perf.index:
    per = per_perf.loc[p]
    mod = mod_perf[(mod_perf.beach == per.beach) &
                   (mod_perf.FIB == per.FIB) &
                   (mod_perf.partition == per.partition) &
                   (mod_perf.lead_time == per.lead_time) &
                   (mod_perf.dataset == per.dataset)]
    if len(mod) == 0:
        continue
    
    temp = mod[index_cols].copy()
    for met in mets:
        temp['diff_'+met] = mod[met] - per[met]
        temp['imp_'+met] = (temp['diff_' +met] > 0).astype(int)
        temp.loc[temp['diff_' +met].isna(),'imp_'+met] = np.nan
    
    per_compare = per_compare.append(temp)

per_compare = per_compare[per_compare.lead_time != 0]  # remove nowcast

# mean difference from persistence
per_compare.groupby(['beach', 'FIB','lead_time','model']).mean()[['diff_' + m for m in mets]].round(2) 

per_compare.groupby(['model']).describe()[['diff_' + m for m in mets]].round(2) # by model
per_compare.describe()[['diff_' + m for m in mets]].round(2)# overall

# number improved over persistence by model
per_compare.groupby(['beach', 'FIB','model']).sum()[['imp_'+m for m in mets]] 
(per_compare.groupby(['lead_time', 'model']).sum()[['imp_'+m for m in mets]] / (2*2*n_part)).round(2)
(per_compare.groupby(['model']).sum()[['imp_'+m for m in mets]] / (n_lead*2*2*n_part)).round(2)
per_compare.sum()[['imp_'+m for m in mets]] / (2*2*n_part*n_lead*n_model) # overall

## Proportion of partitions with at least one model > PER
cases = per_compare.groupby(['beach','FIB','partition','lead_time']).count().index # cases (beach/FIB/partition/lead_time)
imp_per = pd.DataFrame()
for c in cases:
    case = per_compare.loc[(df_perf.beach==c[0]) &
                       (df_perf.FIB == c[1]) &
                       (df_perf.partition == c[2]) &
                       (df_perf.lead_time == c[3])]
    
    temp = pd.DataFrame(index=[c], columns=['imp_'+m for m in mets])
    for m in ['imp_'+m for m in mets]:
        imp = (case[m].sum() > 0).astype(int)
        temp[m] = imp
    imp_per = imp_per.append(temp)
imp_per.index = cases
imp_per.reset_index(inplace=True)

## % cases where at least one model was better than PER
(imp_per.sum()[['imp_'+m for m in mets]] / len(imp_per))  # Overall
(imp_per.groupby(['beach','FIB','lead_time']).sum()[['imp_'+m for m in mets]]) / n_part # Beach/FIB

## Stats Test - Models significantly better than PER?

# Wilcoxon SIgned Rank  Paired test
print('\nWilcoxon Signed Rank Tests:')
print('(p-value for testing if model perf greater than PER')

print('\nBy model')
for i in mets:
    print(i)
    for m in model_map.values():
        if m == 'PER':
            continue
        w, p = stats.wilcoxon(df_for[df_for.model == m][i], df_for[df_for.model == 'PER'][i], alternative='greater')
        print(m + ' - ' + str(w) + '/' + str(round(p,3)))
    print('\n')


## Boxplots - difference from persistence
# sns.catplot(x='lead_time',y='value', hue='model',col='variable', #row = 'FIB',
#             data=per_compare.melt(id_vars = ['lead_time','model','FIB'], value_vars = ['diff_'+m for m in mets]),
#             kind='box')
# plt.tight_layout()

## Bar plots - proportion improved over persistence
# sns.catplot(x='lead_time',y='value', hue='model',col='variable', #row = 'FIB',
#             data=per_compare.melt(id_vars = ['lead_time','model','FIB'], value_vars = ['imp_'+m for m in mets]),
#             kind='bar')
# plt.tight_layout()



### Compared to Nowcast
now_compare = pd.DataFrame()
now_perf = df_perf[df_perf.lead_time == 0]
fc_perf = df_perf[df_perf.lead_time != 0]

for n in now_perf.index:
    now = now_perf.loc[n]
    fc = fc_perf[(fc_perf.beach == now.beach) &
                   (fc_perf.FIB == now.FIB) &
                   (fc_perf.partition == now.partition) &
                   (fc_perf.dataset == now.dataset) &
                   (fc_perf.model == now.model)]
    if len(fc) == 0:
        continue
    
    assert (fc.N == now.N).sum() == len(fc.N), 'nowcast and forecast N not equivalent' # check N/exc same
    assert (fc.exc == now.exc).sum() == len(fc.exc), 'nowcast and forecast N not equivalent'
    
    temp = fc[index_cols].copy()
    for met in mets:
        temp['diff_'+met] = fc[met] - now[met]
        temp['imp_'+met] = (temp['diff_' +met] > 0).astype(int)
        temp.loc[temp['diff_' +met].isna(),'imp_'+met] = np.nan
    
    now_compare = now_compare.append(temp)

## Nowcast performance
now_perf.groupby(['lead_time']).describe()[mets].loc[:, pd.IndexSlice[:,['50%','25%', '75%']]].round(2)
df_perf[df_perf.model == 'PER'].groupby(['lead_time']).describe()[mets].loc[:, pd.IndexSlice[:,['50%','25%', '75%']]].round(2).loc[0]# PER performance on LT=0

## Mean difference from NowCast by beach and FIB and LT
now_compare[now_compare.model !='PER'].groupby(['lead_time']).describe()[['diff_' + m for m in mets]].loc[:, pd.IndexSlice[:,['50%','25%', '75%']]].round(3)
now_compare[now_compare.model !='PER'].groupby(['beach','FIB','lead_time']).mean()[['diff_' + m for m in mets]].round(3)


# Wilcoxon SIgned Rank  Paired test
print('\nWilcoxon Signed Rank Tests:')
print('(p-value for testing if model perf leass than NowCast')

print('\nOverall (N = ' + str(len(beach)*len(fib)*n_part*n_model) + ' per group)')
for i in mets:
    print(i)
    temp = df_perf[df_perf.model != 'PER']
    for l in [1,2,3]:
            w, p = stats.wilcoxon(temp[temp.lead_time == 0][i], temp[temp.lead_time == l][i], alternative='greater')
            print('LT: ' + str(l) + ' - ' + str(w) + '/' + str(round(p,3)))
    print('\n')

print('\nBy model')
for i in mets:
    print(i)
    for m in models:
        if m == 'PER':
            continue
        print(m)
        temp = df_perf[df_perf.model == m]
        for l in [1,2,3]:
            w, p = stats.wilcoxon(temp[temp.lead_time == 0][i], temp[temp.lead_time == l][i], alternative='greater')
            print('LT: ' + str(l) + ' - ' + str(w) + '/' + str(round(p,3)))
        print('\n')
    print('\n')


# sns.catplot(x='lead_time',y='value', hue='model',col='variable', #row = 'FIB',
#             data=now_compare.melt(id_vars = ['lead_time','model','FIB'], value_vars = ['diff_'+m for m in mets]),
#             kind='box')
# plt.tight_layout()
