#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the function 'beach_model' using given inputs
- Can be used to perform a single model run OR through multiple cases

Created on Thu Jun 30 13:03:42 2022

@author: rtsearcy
"""

import os
import pandas as pd
import time
import beach_model

# %% Inputs
single_run = True
base_folder = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting'

### Single Model Run
if single_run:
    beach = 'HSB3N'                         # Cowell, Manhattan
    fib = 'ENT'                               # FIB to model: ENT, EC
    
    train_years = list(range(2012, 2020))    # Which years to include in the training set? Note: range doesn't include last number
    test_years = [2020, 2021] #[2020,2021]                 # Typically the following years
    n_partitions = 1
    
    lead_time = [0]#, 2]                      # Lead time; Days ahead to forecast FIB (0 = nowcast)
    
    model_types = ['per','lm','gbm','rf']        # MLR/Binary logistic regression, Random Forest, Support Vector Machine, Gradient Boosted Machine

### Run Cases (if single_run == 0)
else:
    '''
    Iterate through all test cases:

    2 beaches + 2 FIB types + ~10-12 data partitions +
    4 lead times + 4 model types = > 500 cases
    '''
    beach = ['HSB3N']#, 'Manhattan']
    fib = ['EC'] #,'ENT']
    
    lead_time = [0, 1, 2, 3]  # Lead time; Days ahead to forecast FIB (0 = nowcast) 
    model_types = ['rf'] #['per', 'lm', 'svm','gbm']  

    # train/test year breakdowns
    year_start = 2007
    year_end = 2021
    n_train = 6  # number of years in training set
    n_test = 2   # number of years in test set

    train_years = [list(range(year_start + i, year_start + i + n_train)) for
                   i in range(year_end - year_start - n_train - n_test + 2)]
    test_years = [list(range(i[-1]+1, i[-1]+1 + n_test)) for i in train_years]
    n_partitions = len(test_years)
    '''N data partitions = year_end - year_start - n_train - n_test + 2 '''

# %% Iterate

'''Run through all beach, FIB, data partitions, forecast lead times,
and models'''

# Check Single Run Parameters
if type(beach) == str:
    beach = [beach]
if type(fib) == str:
    fib = [fib]
if type(train_years[0]) == int:
    train_years = [train_years]
if type(test_years[0]) == int:
    test_years = [test_years]
if type(lead_time) == int:
    lead_time = [lead_time]
if type(model_types) == str:
    model_types = [model_types]

# Iterate
s_time = time.time()
for b in beach:
    for f in fib:
        for t in range(0, n_partitions):
            print('- - ' + b + ' / ' + f + ' - -')
            print('\nTrain Years:')
            print(train_years[t])
            print('Test Years:')
            print(test_years[t])

            load_output = beach_model.load_data(b, f,
                                                train_years[t], test_years[t])
            y_train = load_output[0]
            X_train = load_output[1]
            y_test = load_output[2]
            X_test = load_output[3]
            persist = load_output[4]
            drop_list = load_output[5]
            drop_missing = load_output[6]
            partition_folder = load_output[7]

            for lt in lead_time:
                for m in model_types:
                    print('\n+ + Lead Time: ' + str(lt) + ' day(s) + +')
                    if m == 'per':  # Persistence
                        beach_model.model(y_train, persist, y_test, persist, f,
                                          lt, m, partition_folder, save=True)
                    else:
                        beach_model.model(y_train, X_train, y_test, X_test, f,
                                          lt, m, partition_folder, save=True)


print('\nelapsed: ' + str(round(time.time() - s_time, 6)))
