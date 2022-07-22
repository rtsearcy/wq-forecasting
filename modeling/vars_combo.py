#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vars_combo.py
Created on Wed Jan  8 19:37:57 2020

@author: rtsearcy

Update 2022

Combine FIB and EV datasets into a single dataframe for a given beach
- Calculate beach-specific variables

"""

import pandas as pd
import os
import numpy as np
from numpy import sin, cos, pi, isnan, nan, log10

#%% Inputs + Load Data
folder = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting'
beach = 'HSB3N'

sd = '2000-01-01'  # dates of dataset
ed = '2021-12-31'

FIB = {'EC':400, 
       'ENT': 104}

### Lag vars list
lag_range = range(1,6)
lag_vars = [
            'tide_max','tide_min','tide_range',
            'logflow', 'upwell','upwell_bin',
            'WVHT','APD','DPD','wtemp_b','WVHT_q75','DPD_q75',
            'logchl','logturb','cond','DO','wtemp','chl_q75','turb_q75',
            'along','cross','cross_bin','along_bin','current_mag','along_mag','cross_mag',
            'atemp','atemp_min','dtemp','rad','relhum',
            'wspd','gust','awind','owind','wspd_q75', 'owind_bin', 'awind_bin'
             ]
    

### FIB Data
fib_file = os.path.join(folder, 
                        'beach', 
                        beach.replace(' ','_'), 
                        'variables', 
                        'FIB_variables_' + beach.replace(' ','_') + '.csv')
fib = pd.read_csv(fib_file, index_col=['date'], parse_dates=['date'])
fib.drop(['sample_time'], axis=1, inplace=True)

fib['year'] = fib.index.year
fib['month'] = fib.index.month
fib['doy'] = fib.index.dayofyear # Day of Year
fib['dow'] = fib.index.dayofweek # Monday=0, Sunday=6
fib['weekend'] = (fib.dow.isin([6,5,4])).astype(int)
fib['weekend1'] = (fib.dow.isin([0,6,5])).astype(int)

print(beach)
print('\nFIB (N=' + str(len(fib)) + ', ' + str(fib.index.year.min()) + '-' + str(fib.index.year.max()) + ')\nMissing:')
print(fib.isna().sum().sort_values(ascending=False).replace(0,np.nan).dropna())

### Load Enviro. Variables
stations = pd.read_csv(os.path.join(folder,'beach','beach_env_stations.csv'), index_col='beach')
stations = stations.loc[beach]
angle = stations['angle']  # Cowell - 135, MB - 

## Flow
try:
    s = [s for s in os.listdir(os.path.join(folder,'data/flow/')) if stations.flow.replace(' ','_') in s][0]
    flow = pd.read_csv(os.path.join(folder,'data/flow/',s), parse_dates=['date'], index_col=['date'])

    print('\nFlow (N=' + str(len(flow)) + ', ' + str(flow.index.year.min()) + '-' + str(flow.index.year.max()) + ')\nMissing:')
    print(flow.isna().sum().sort_values(ascending=False).replace(0,np.nan).dropna())
except:
    flow = pd.DataFrame()

## Tide
s = [s for s in os.listdir(os.path.join(folder,'data/tide/')) if stations.tide.replace(' ','_') in s][0]
tide = pd.read_csv(os.path.join(folder,'data/tide/',s), parse_dates=['date'], index_col=['date'])

print('\nTide (N=' + str(len(tide)) + ', ' + str(tide.index.year.min()) + '-' + str(tide.index.year.max()) + ')\nMissing:')
print(tide.isna().sum().sort_values(ascending=False).replace(0,np.nan).dropna())

## Upwelling
s = [s for s in os.listdir(os.path.join(folder,'data/upwelling/')) if stations.upwell.replace(' ','_') in s][0]
upwell = pd.read_csv(os.path.join(folder,'data/upwelling/',s), parse_dates=['date'], index_col=['date'])

print('\nUpwelling (N=' + str(len(upwell)) + ', ' + str(upwell.index.year.min()) + '-' + str(upwell.index.year.max()) + ')\nMissing:')
print(upwell.isna().sum().sort_values(ascending=False).replace(0,np.nan).dropna())

## Waves
s = [s for s in os.listdir(os.path.join(folder,'data/waves/')) if stations.waves.replace(' ','_') in s][0]
wave = pd.read_csv(os.path.join(folder,'data/waves/',s), parse_dates=['dt'], index_col=['dt'])
wave.index.name = 'date'

wave['WVHT_q75'] = (wave.WVHT > wave.WVHT.quantile(.75)).astype(int)
wave.loc[wave.WVHT.isna(),'WVHT_q75'] = np.nan
wave['DPD_q75'] = (wave.DPD > wave.DPD.quantile(.75)).astype(int)
wave.loc[wave.DPD.isna(),'DPD_q75'] = np.nan

print('\nWave (N=' + str(len(wave)) + ', ' + str(wave.index.year.min()) + '-' + str(wave.index.year.max()) + ')\nMissing:')
print(wave.isna().sum().sort_values(ascending=False).replace(0,np.nan).dropna())

## Water Quality
s = [s for s in os.listdir(os.path.join(folder,'data/water_quality/CenCOOS')) if stations.wq.replace(' ','_') in s][0]
wq = pd.read_csv(os.path.join(folder,'data/water_quality/CenCOOS',s), index_col=['dt'], parse_dates=['dt'])
wq.index = wq.index.date
wq.index = wq.index.astype('datetime64[ns]')
wq.index.name = 'date'

for c in ['chl','turb']:
    if c not in wq.columns:
        continue
    wq['log'+c] = np.log10(wq[c]+1)
    wq[c+'_q75'] = (wq[c] > wq[c].quantile(.75)).astype(int)
    wq.loc[wq[c].isna(),c+'_q75'] = np.nan

print('\nWater Quality (N=' + str(len(wq)) + ', ' + str(wq.index.year.min()) + '-' + str(wq.index.year.max()) + ')\nMissing:')
print(wq.isna().sum().sort_values(ascending=False).replace(0,np.nan).dropna())

## Currents
try:
    s = [s for s in os.listdir(os.path.join(folder,'data/currents/')) if stations.currents.replace(' ','_') in s][0]
    currents = pd.read_csv(os.path.join(folder,'data/currents/',s), index_col=['dt'], parse_dates=['dt'])
    currents.index.name = 'date'
    
    currents.drop(['lat','lon'], axis=1, inplace=True)
    
    currents['current_mag'] = np.sqrt((currents.u**2) + (currents.v**2))
    currents['current_q75'] = (currents['current_mag'] > currents['current_mag'].quantile(.75)).astype(int)
    currents.loc[currents['current_mag'].isna(),'current_q75'] = np.nan
    
    currents['cross_bin'] = np.nan
    currents.loc[currents.cross > 0,'cross_bin'] = 1
    currents.loc[currents.cross < 0,'cross_bin'] = 0
    
    currents['cross_mag'] = currents.cross.abs()
    
    currents['along_bin'] = np.nan
    currents.loc[currents.along > 0,'along_bin'] = 1
    currents.loc[currents.along < 0,'along_bin'] = 0
    
    currents['along_mag'] = currents.along.abs()
    
    print('\nCurrents (N=' + str(len(currents)) + ', ' + str(currents.index.year.min()) + '-' + str(currents.index.year.max()) + ')\nMissing:')
    print(currents.isna().sum().sort_values(ascending=False).replace(0,np.nan).dropna())
    
except:
    currents = pd.DataFrame()

## Met
met = pd.read_csv(stations.met, parse_dates=['date'], index_col=['date'])

# Note: CIMIS data do not include wind direction...taking wind from other station
wind = pd.read_csv(stations.wind, parse_dates=['dt'], index_col=['dt'])# download hourly data
wind['awind'] = wind['wspd'] * round(np.sin(((wind['wdir'] - angle) / 180) * np.pi), 1)
wind['owind'] = wind['wspd'] * round(np.cos(((wind['wdir'] - angle) / 180) * np.pi), 1)
wind = wind.resample('1D').mean()
wind.index = wind.index.date
wind.index.name = 'date'
wcols = ['gust','wspd','awind','owind']
wind = wind[[c for c in wcols if c in wind.columns]]
wcols = wind.columns
# for i in [1,2,3]:
#     wind[[c+str(i) for c in wcols]] = wind[wcols].shift(i) # lags

met.drop([c for c in wcols if c in met.columns],axis=1, inplace=True)
met = pd.concat([met,wind], axis=1)

met['wet'] = ((met['rain'] + met['rain3T']) > 0.1*25.4).astype(int)   # greater than .1 in (in mm)
    
met['wspd_q75'] = (met.wspd > met.wspd.quantile(.75)).astype(int)
met.loc[met['wspd'].isna(),'wspd_q75'] = np.nan
met['owind_bin'] = (met.owind > 0).astype(int)
met.loc[met['owind'].isna(),'owind_bin'] = np.nan
met['awind_bin'] = (met.awind > 0).astype(int)
met.loc[met['awind'].isna(),'awind_bin'] = np.nan

print('\nMet (N=' + str(len(met)) + ', ' + str(met.index.year.min()) + '-' + str(met.index.year.max()) + ')\nMissing:')
print(met.isna().sum().sort_values(ascending=False).replace(0,np.nan).dropna())


### Combine into ENV dataframe
env = pd.concat([met, wq, tide, wave, upwell, currents, flow], axis=1)

#%% Lag Variables
assert len(env) == len(pd.date_range(env.index[0],env.index[-1])), 'EVs not continuous time series'

### EVs (except rain)
for v in lag_vars:
    if v not in env:
        continue
    
    for i in lag_range:
        if v[-1].isalpha():
            var_name = v+str(i)
        else:
            var_name = v + '_' + str(i)
        
        env[var_name] = env[v].shift(i)
        
### Rain (shift-lag totals)
# Rain totals between "lag_start" and "lag_start"+"lag_shift" days ago
lag_start = range(1,6)  
lag_shift = [2,3,5,7,14,30]

rain = met.rain.copy()
assert len(rain) == len(pd.date_range(rain.index[0],rain.index[-1])), 'rain df not continuous time series'

for i in lag_start:
    for s in lag_shift:
        var_name = 'lograin' + str(i) + '_' + str(i+s-1) + 'T'
        
        temp = rain.copy().shift(i)
        for j in range(1,s):
            temp += rain.copy().shift(i+j)
        
        env[var_name] = np.log10(temp + 1)
        
    
#%% Combine and Save
#df = pd.merge(fib, env, how='left', left_index=True, right_index=True)  # index of FIB obs days only
df = pd.concat([fib, env], axis=1)   # index is ALL days in range
df.sort_index(ascending=True, inplace=True)
df = df[sd:ed]

out_file = os.path.join(folder, 'beach', beach.replace(' ','_'), 'variables', beach.replace(' ','_') + '__variables.csv')
df.to_csv(os.path.join(out_file))

print('Beach - ' + beach)
print('N - ' + str(len(df)))
print('Cols - ' + str(len(df.columns)) +'\n')
print(df.columns)
