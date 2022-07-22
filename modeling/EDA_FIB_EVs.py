#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various Exploratory Data Analyses (EDA) for FIB and EVs

Created on Tue Feb 15 09:24:55 2022

@author: rtsearcy
"""

import pandas as pd
import os
import datetime
import numpy as np
from scipy import stats
from scipy.stats.mstats import gmean
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

from sklearn.metrics import mutual_info_score as MI
from sklearn.feature_selection import mutual_info_regression as MIR

#%% Load Data + Plot Parameters
#folder = '/Users/rtsearcy/Box/water_quality_modeling/forecasting/'
folder = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting'
beach = 'Cowell'
#beach = 'HSB3N'

FIB = {'EC':400, 
       'ENT': 104}

infile = os.path.join(folder, 'beach', beach.replace(' ','_'), 'variables', beach.replace(' ','_') + '__variables.csv')
df = pd.read_csv(infile, parse_dates=['date'], index_col = ['date'])

df = df['2007':'2021']
df['weekend'] = (df.index.dayofweek > 3).astype(int)

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


#%% FIB - Times Series + Stats
df_fib = df[[c for c in df.columns if ('ENT' in c) or ('EC' in c)]].copy()#.dropna()
df_fib['year'] = df_fib.index.year
df_fib['month'] = df_fib.index.month
df_fib['doy'] = df_fib.index.dayofyear # Day of Year

fib_colors = {'EC':'k',
              'ENT':'r'}

### Time Series 
# plt.figure(figsize = (10,6))
# c = 1
# for f in FIB:
#     #p = plt.plot('log'+f, data=df_fib, label=f)
    
#     # p = plt.bar(df_fib.index, df_fib['log'+f] + 0.5, bottom = -.5, width=1)
#     # plt.scatter(df_fib.index, df_fib['log'+f], color='k', s=5)
    
#     #plt.axhline(np.log10(FIB[f]),ls=":",color=p[0].get_color(),lw=1)
    
#     plt.subplot(len(FIB),1,c)
#     for y in df_fib.year.unique():
#         temp = df_fib[df_fib.year == y].copy()
#         plt.plot(temp.doy, temp['log'+f], ls='-', marker='o', label = str(y)) #,color=fib_colors[f])
#     plt.axhline(np.log10(FIB[f]),ls=":",color='k',lw=1)
#     plt.xticks(ticks = [91,121,152,182,213,244,274], labels = ['Apr','May','Jun','Jul','Aug','Sep','Oct'])
#     plt.title(f)
#     c+=1
# #plt.legend()
# plt.suptitle(beach)
# plt.tight_layout()

### Recent time series
plt.figure(figsize = (12,12))
years = [2017,2018,2019,2020,2021]
c = 1
for y in years:
    
    for f in FIB:
        plt.subplot(len(years),len(FIB),c)
        p = plt.plot('log'+f, data=df_fib[str(y)], color='k', ls='-',marker='.')
        plt.axhline(np.log10(FIB[f]),ls=":",color='k', alpha=.7,lw=1)
        
        if c < 3:
            plt.title(f)
            
        if c%2 > 0:
            plt.ylabel('log10[MPN/100mL]')
        
        c+=1
plt.suptitle(beach)
plt.tight_layout()


### FIB Boxplots by Year and Month
plt.figure(figsize=(10,6))
c=1
for t in ['month','year']:
    plt.subplot(2,1,c)
    temp = df_fib.melt(id_vars=t, value_vars=['logEC','logENT'], var_name='FIB', value_name='conc')
    sns.boxplot(x=t, y='conc',hue='FIB',data=temp)
    plt.ylabel('log10 CFU/100 mL')
    
    c+=1
plt.suptitle(beach)
plt.tight_layout()

### Data summary
fib_sum = pd.DataFrame()
for f in FIB.keys():
    temp = df_fib[[f+'_quant',f+'_exc']].copy()
    temp['status'] = np.nan
    temp[f] = 1
    temp.loc[temp[f+'_quant']==0,'status'] = 'Below LOQ'
    temp.loc[(temp[f+'_quant']==1) & (temp[f+'_exc']==0),'status'] = 'LOQ < FIB < SSS'
    temp.loc[temp[f+'_exc']==1,'status'] = 'Exceed SSS'
    temp.drop([f+'_quant',f+'_exc'],axis=1, inplace=True)
    temp = temp.groupby(['status']).count()
    fib_sum = pd.concat([fib_sum,temp], axis=1)

print(beach)
print(df_fib.dropna().index[0])
print(df_fib.dropna().index[-1])
print(fib_sum.sum())
print(fib_sum.iloc[[0,2,1]])


#%% Autocorrelation Analysis - EVs

evs = [
       'tide','tide_range','tide_max',
       'logchl','logturb',
       'wtemp_b','WVHT','APD', 'DPD',
       'owind','awind','dtemp','atemp','rad', 'rain',
       'logflow',
       'along','cross'
       ]

evs = [e for e in evs if e in df.columns]

lags = range(1,8)
sig_p = 0.05
df_rho = pd.DataFrame(index=[evs],columns = list(lags))
df_p = df_rho.copy()

## Autocorrelation
for e in evs:
    if e not in df.columns:
        continue
    #print(e)
    #p = acf(df[e].dropna(), nlags=len(lags)-1)  # pearson correlation between t and t-lag
    #p = [round(p,3) for p in p]
    
    for i in lags:        
        temp = pd.concat([df[e],df[e].shift(i)],axis=1).dropna()
        temp.columns = ['base', 'lag']
        rho, p = stats.spearmanr(temp.base, temp.lag)
        #rho = round(temp.corr(method='spearmant').iloc[0,1],3)
        df_rho.loc[e,i] = rho
        df_p.loc[e,i] = p


plt.figure(figsize=(7,5))

df_annot = df_rho.astype(float).round(2).astype(str) + (df_p < 0.05).astype(int).replace([0,1],['','*']).astype(str)
print(beach)
print(df_annot)

cmap = sns.light_palette('#444444',as_cmap = True)
sns.heatmap(df_rho.astype(float), annot=df_annot, fmt='s', linecolor='k', linewidths=.25,
            cmap= cmap, 
            cbar_kws={'label':'Spearman Rank Correlation'}, vmin=-0.5, vmax=1, robust=True) # YlGnBu_r,


plt.xlabel('Lag')
plt.ylabel('')
plt.title(beach_name + ' (' + abbrev + ')')

plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False)         # ticks along the top edge are off

plt.tight_layout()


#%% EV Cross-Corr w FIB

evs = [
       'tide','tide_range','tide_max',
       'logchl','logturb',
       'wtemp_b','WVHT','APD', 'DPD',
       'owind','awind','dtemp','atemp','rad', 
       'rain','rain3T',
       'logflow',
       'along','cross'
       ]

evs = [e for e in evs if e in df.columns]

lags = range(0,8)
sig_p = 0.05

for f in FIB:
    df_rho = pd.DataFrame(index=[evs],columns = list(lags))
    df_p = df_rho.copy()
    
    for e in evs: 
        for i in lags:        
            temp = pd.concat([df[f],df[e].shift(i)],axis=1).dropna()
            rho, p = stats.spearmanr(temp[f], temp[e])
            #rho = round(temp.corr(method='spearmant').iloc[0,1],3)
            df_rho.loc[e,i] = rho
            df_p.loc[e,i] = p
    
    plt.figure(figsize=(7,5))
    
    df_annot = df_rho.astype(float).round(2).astype(str) + (df_p < 0.05).astype(int).replace([0,1],['','*']).astype(str)
    print(beach)
    print(df_annot)

    cmap = sns.diverging_palette(220, 20, as_cmap=True) 
    sns.heatmap(df_rho.astype(float), annot=df_annot, fmt='s',
                linecolor='k', linewidths=.25,
                cmap= cmap, 
                cbar_kws={'label':'Spearman Rank Correlation'},
                vmin=-0.3,vmax=0.3, robust=False, center=0) # YlGnBu_r
    plt.xlabel('Lag')
    plt.ylabel('')
    plt.title(beach_name + ' (' + abbrev + ') - ' + f)

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left=False) 
    
    plt.tight_layout()


#%% Categorical EVs

quant_only = False ## consider all data or just quantified

cat_ev = [
           'tide_spring',
           'WVHT_q75','DPD_q75',
           'chl_q75', 'turb_q75',
           'wspd_q75','awind_bin','owind_bin',
           'wet',
           'along_bin','cross_bin', 'current_q75',
           'flow1_q75', 'flow1_q90',
           'weekend', 'month'
           ]

df_cat = df.copy()
if quant_only:
    for f in FIB:
        df_cat.loc[df_cat[f+'_quant']==0,'log'+f] = np.nan

print('Sig diff between var states in FIB ?(Kruskal-Wallis p-vals)')
for v in cat_ev:
    if v not in df.columns:
        continue
    
    temp = df_cat.melt(id_vars=v, value_vars=['logEC','logENT'], var_name='FIB', value_name='conc').dropna()
    
    ### Boxplots w FIB_exc, FIB_quant
    # plt.figure(figsize = (4,4))
    
    # #plt.subplot(1,3,1)  # QUANT
    # sns.boxplot(x=v,y='conc', hue='FIB', data = temp)
    # plt.ylabel('logFIB')
    # plt.xlabel(v)
    # #plt.xticks(ticks=[0,1], labels=['BLOQ', 'QUANT'])
    # #plt.legend([], frameon=False)
    
    # plt.tight_layout()
    
    ### Hypothesis Testing (EV Regimes)
    print('\n'+v + ' (N=' + str(len(temp)/2) + ')')
    for f in FIB:
        kw_temp = temp[temp.FIB == 'log'+f]
        if v != 'month':
            kw = stats.kruskal(kw_temp[kw_temp[v]==0]['conc'],kw_temp[kw_temp[v]==1]['conc'])
        else:
            kw = stats.kruskal(kw_temp[kw_temp[v]==4]['conc'],
                               kw_temp[kw_temp[v]==5]['conc'],
                               kw_temp[kw_temp[v]==6]['conc'],
                               kw_temp[kw_temp[v]==7]['conc'],
                               kw_temp[kw_temp[v]==8]['conc'],
                               kw_temp[kw_temp[v]==9]['conc'],
                               kw_temp[kw_temp[v]==10]['conc'])
        
        print('\n  ' + f + ' - ' + str(round(kw[1],3)))
        print('mean/med:')
        print(kw_temp.groupby([v]).describe().round(2)['conc'][['mean','50%']])

#%% OLD

###Top Correlations

quant_only = False ## consider all data or just quantified
lag = 0

### Spearman Correlations 
# p vals
print(beach + ' - Spearman Correlations')
for f in FIB:
    if quant_only:
        df_corr = df[df[f+'_quant']==1]
        print('\n*** QUANTIFIED SAMPLES ONLY ***')
    else:
        df_corr = df.copy()
    
    corr_vars = [var for var in df_corr.columns if (f not in var) and (var not in ['sample_time'])]
    fib_vars = [var for var in df_corr.columns if var not in corr_vars]
    
    if lag > 0:
        print('+++ LAG ' + str(lag) + ' EVs +++')
        fib_temp = df[fib_vars].copy()
        ev_temp = df[corr_vars].copy().shift(lag)
        df_corr = pd.concat([fib_temp,ev_temp], axis=1)
    
    ## Scipy
    corr = pd.DataFrame()
    for c in corr_vars:
        r = stats.spearmanr(df_corr[[f,c]].dropna())
        temp = pd.DataFrame({'rho':r[0],
                             'p-val':r[1],
                             'N': len(df_corr[[f,c]].dropna())
                             }, index=[c])
        corr = corr.append(temp)
    #corr.index.name = 'var'
    corr['rho'] = corr.rho.round(3)
    corr = corr.sort_values('p-val')
    
    # ## Pandas
    # corr = df_corr.corr(method='spearman')[f]
    # corr = corr.loc[corr_vars]#.sort_values()
    # corr.name = 'rho'
    # corr = corr.to_frame()
    # corr['direction'] = '+'
    # corr.loc[corr.rho < 0,'direction'] = '-'
    # corr['rho'] = corr.rho.abs().round(3)
    # corr = corr.sort_values('rho', ascending=False)
    
    print('\n'+f)
    print(corr.head(20))
    print('\n')
    print(corr.tail(20))

### Correlation by Year
df_corr = df.copy()
years = range(2010,2022)

fib_color = {'EC':'k',
              'ENT':'r'}

corr_vars = [
    'dtemp',
    'tide',
    'rad',
    'month','dow',
    'wtemp_b',
    'chl','turb',
    'DPD',
    'logflow',
    'rain3T',
    'cross'
    ]

for c in corr_vars:
    if c not in df_corr.columns:
        continue
    plt.figure(figsize = (8,4))
    for f in FIB:
        corr_year = []
        for y in years:
            temp = df_corr[str(y)].copy()
            r = stats.spearmanr(temp[[f,c]].dropna())
            corr_year += [r[0]]
        plt.plot(years,corr_year, ls=':', marker='.', color = fib_color[f], label=f)
        
        r_avg = stats.spearmanr(df_corr[str(years[0]):str(years[-1])][[f,c]].dropna())[0]
        plt.axhline(r_avg, ls='-.', color = fib_color[f], alpha=0.75)# plot avg. corr
        
    plt.axhline(0, color='grey')
    plt.ylabel('Spearman Correlation')
    plt.title(c)
    plt.legend(frameon=False)

### Mutual Information

quant_only = False ## consider all data or just quantified
lag = 0

warnings.filterwarnings("ignore")
print('\n' + beach + ' - Mutual Information')
for f in FIB:
    if quant_only:
        df_mut = df[df[f+'_quant']==1]
        print('*** QUANTIFIED SAMPLES ONLY ***')
    else:
        df_mut = df.copy()
    
    corr_vars = [var for var in df_mut.columns if (f not in var) and (var not in ['sample_time'])]
    fib_vars = [var for var in df_mut.columns if var not in corr_vars]
    
    if lag > 0:
        print('+++ LAG ' + str(lag) + ' EVs +++')
        fib_temp = df[fib_vars].copy()
        ev_temp = df[corr_vars].copy().shift(lag)
        df_mut = pd.concat([fib_temp,ev_temp], axis=1)
    
    mut = pd.DataFrame()
    for c in corr_vars:
        temp = df_mut[[f,c]].dropna()
        m = MI(temp[f],temp[c])
        out = pd.DataFrame({'mut_info':round(m,3),
                             #'p-val':r[1],
                             'N': len(temp)
                             }, index=[c])
        mut = mut.append(out)
    mut = mut.sort_values('mut_info', ascending=False)
    
    print('\n'+f)
    print(mut.head(15))
    print('\n')
    print(mut.tail(15))
    
#warnings.filterwarnings('always')


### Continuous EVs

cont_ev = [
           'tide',#'tide_max','tide_range',
           'lograin1','lograin7T',#'rain30T',
           'rad',
           'temp','dtemp','relhum',
           'wspd','gust','awind','owind',
           'wtemp',
           'WVHT','DPD',
           'upwell','logchl','cond','logturb',
           'along_mag','cross_mag',
           ]

df_cont = df.copy()
# if quant_only:
#     for f in FIB:
#         df_cont.loc[df_cont[f+'_quant']==0,'log'+f] = np.nan

for v in cont_ev:
    if v not in df.columns:
        continue
    
    plt.figure(figsize = (9,4))
    
    ### Scatter Plots w logFIB
    temp = df_cont.melt(id_vars=v, value_vars=['logEC','logENT'], var_name='FIB', value_name='conc')
    
    plt.subplot(1,3,1)
    sns.scatterplot(x=v, y='conc',hue='FIB', data=temp)
    plt.ylabel('log10 CFU/100 mL')
    plt.xlabel(v)
    plt.title(v + ' (' + beach + ', N='+ str(len(temp)/2) +')')
    
    
    ### Boxplots w FIB_exc, FIB_quant
    temp = df_cont.melt(id_vars=v, value_vars=['EC_quant','ENT_quant'], var_name='FIB', value_name='quant').dropna()
    
    plt.subplot(1,3,2)  # QUANT
    sns.boxplot(x='quant',y=v, hue='FIB', data = temp)
    plt.ylabel(v)
    plt.xlabel('')
    plt.xticks(ticks=[0,1], labels=['BLOQ', 'QUANT'])
    plt.legend([], frameon=False)
    
    
    ### Boxplots w FIB_exc, FIB_quant
    temp = df_cont.melt(id_vars=v, value_vars=['EC_exc','ENT_exc'], var_name='FIB', value_name='exc').dropna()
    
    plt.subplot(1,3,3)  # EXC
    sns.boxplot(x='exc',y=v, hue='FIB', data = temp)
    plt.ylabel(v)
    plt.xlabel('')
    plt.xticks(ticks=[0,1], labels=['< SSS', '> SSS'])
    plt.legend([], frameon=False)
    
    plt.tight_layout()
    