#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
beach_model.py - Master code; function which creates FIB models using beach-specific
variable datasets.                                                             
RTS - June 2022

Steps:
    1. Accept Inputs: beach, FIB, dataset partition, forecast lead time, model
    2. Load data, remove uneccessary/correlated vars, deal with missingness
    3. Partition into train and test datasets
    4. Select variables / Fit models / Cross-validate to optimize
    (Regularization)
    5. Check performance on train and test sets
    6. Save

Created on Tue Feb 15 09:24:55 2022

@author: rtsearcy
"""
# def beach_model(beach, fib, train_years, test_years, lead_time, model_type, 
#                 base_folder='/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting', 
#                 save=True):
    
### Packages
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold, RFECV, SequentialFeatureSelector, RFE
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, r2_score, roc_auc_score, balanced_accuracy_score
import joblib
from datetime import datetime
import time

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)

### Embedded Inputs
base_folder = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting'

output_bin = True                       # Output: Binary (Exc/Non-Exc) - True; Continuous (logFIB) - False

score_metric = 'balanced_accuracy'      # Binary: f1, roc_auc, recall, precision, accuracy, balanced_accuracy
#score_metric = 'max_error'     # Regress: r2, neg_root_mean_squared_error, max_error

keep_vars = [                            # Variables to always keep despite missingness
    #'logENT_ant',
    #fib + '_ant_exc',
    # 'dow',
    #'current_mag3','along3','cross3',
    ]

check_correl = True                        # Removes highly correlated variables
miss_allow = 0.15                          # fraction of total samples allowed missing before variable is dropped
scale = True                              # Scale each variable by mean and variance

vs_method = 'select'                       # Variable Selection Method 
'''
all - use all variables remaining after cleaning
force - force to select only those in keep_vars
select - use variable selection methods emedded in learning algorithms
'''

## Constants
fib_thresh = {'EC': 400, 'ENT': 104}  

#%% Embedded Functions 
def model_eval(true, predicted, thresh=0.5, output_bin=True):  # Evaluate Model Performance
    # if true.dtype == 'float':
        
    if not output_bin:
        r2 = r2_score(true, predicted)
        rmse = np.sqrt(((true - predicted)**2).mean())

    samples = len(true)  # number of samples
    exc = true.sum()  # number of exceedances
    
    if exc == 0:
        auc = np.nan
    else:
        auc = roc_auc_score(true, predicted)
        
    true = (true > thresh).astype(int)  # Convert to binary if predicted.dtype == 'float':
    predicted = (predicted > thresh).astype(int)

    cm = confusion_matrix(true, predicted)   # Lists number of true positives, true negatives,false pos,and false negs.
    if cm.size == 1: ## No exc observed or predicted
        sens = np.nan
        spec = 1.0
        acc = 1.0
        bal_acc = 1.0
    else: 
        sens = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # sensitivity - TP / TP + FN
        spec = cm[0, 0] / (cm[0, 1] + cm[0, 0])  # specificity - TN / TN + FP
        acc = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])
        bal_acc = balanced_accuracy_score(true, predicted) # balanced acc -> deals with class imbalance
    


    if output_bin:
        out = {'sens': round(sens, 3), 'spec': round(spec, 3), 'acc': round(acc, 3),
               'bal_acc': round(bal_acc, 3), 'AUC': round(auc, 3), 'N': samples, 'exc': exc}
    else:
        out = {'sens': round(sens, 3), 'spec': round(spec, 3), 'acc': round(acc, 3),
               'R2': round(r2, 3), 'RMSE': round(rmse, 3), 'AUC': round(auc, 3),
               'N': samples, 'exc': exc}
 
    return out

def check_corr(dep, ind, thresh=0.9, method='spearman'):
    '''
    Check if confounding variables have correlations > thresh, and drop the one with 
    least correlation to the dependnet variable
    
    Parameters:
        - dep - Pandas Series of the dependant variable
        
        - ind - Dataset (Pandas DF) containing modeling variables to be checked against the dependent
          variables
        
        - thresh - Threshold for Pearson Correlation Coefficient between two variables
          above which the least correlated variable to FIB will be dropped
          
    Output:
        - DataFrame with the best correlated variables included
    
    '''
    print('\nChecking variable correlations against threshold (' + method.capitalize() + ' correlation > ' + str(thresh) + '): ')
    c = ind.corr()  # Pearson correlation coefs.
    to_drop = []

    for ii in c.columns:  # iterate through all variables in correlation matrix except dependant variable
        temp = c.loc[ii]
        temp = temp[temp.abs() > thresh]  # .5 removed a lot of variables
        temp = temp.drop(ii, errors='ignore')  # Remove the variable itself
        i_corr = dep.corr(ind[ii], method=method)
        if len(temp) > 0:
            for j in temp.index:
                j_corr = dep.corr(ind[j],method=method)
                if ii not in to_drop and abs(i_corr) < abs(j_corr):  # Drop variable if its corr. with logFIB is lower
                    to_drop.append(ii)

    print('  Dropped ' + str(len(to_drop)) + ' highly correlated vars')
    #print(to_drop)
    ind = ind.drop(to_drop, axis=1, errors='ignore')  # Drop variables
    #print('  Remaining variables - ' + str(len(ind.columns) - 1))
    #print(ind.columns.values)
    return to_drop

def multicollinearity_check(X, thr):  # Check VIF of model variables, drop if any above 'thr'
    variables = list(X.columns)
    print('\nChecking multicollinearity of ' + str(len(variables)) + ' variables for VIF:')
    if len(variables) > 1:
        vif_model = LinearRegression()
        v = [1 / (1 - (r2_score(X[ix], vif_model.fit(X[variables].drop(ix, axis=1), X[ix]).
                                predict(X[variables].drop(ix, axis=1))))) for ix in variables]
        maxloc = v.index(max(v))  # Drop max VIF var if above 'thr'
        if max(v) > thr:
            print(' Dropped: ' + X[variables].columns[maxloc] + ' (VIF - ' + str(round(max(v), 3)) + ')')
            variables.pop(maxloc)  # remove variable with maximum VIF
        else:
            print('VIFs for all variables less than ' + str(thr))
        X = X[[i for i in variables]]
        return X
    else:
        return X

def drop_lead_time_vars(df, lead_time):
    ''' Drop variables not available for a given lead time '''
    
    drop_list = []
    
    if lead_time >= 1:
        drop_lag1 = [v for v in df.columns if ('1' in v) and ('tide' not in v)  and ('T' not in v)] + ['WVHT1','WVHT_q75_1']
        drop_rain_totals1 = ['lograin2T', 'lograin3T', 'lograin4T', 'lograin5T', 
                            'lograin6T', 'lograin7T', 'lograin14T', 'lograin30T'] + \
                            [v for v in df.columns if 'lograin1_' in v]
        drop_list += drop_lag1 + drop_rain_totals1
        
    if lead_time >= 2:
        drop_lag2 = [v for v in df.columns if ('2' in v) and ('tide' not in v) and ('T' not in v)] + ['WVHT2','WVHT_q75_2']
        drop_rain_totals2 = [v for v in df.columns if 'lograin2_' in v]
        drop_list += drop_lag2 + drop_rain_totals2
        
    if lead_time == 3:
        drop_lag3 = [v for v in df.columns if ('3' in v) and ('tide' not in v) and ('T' not in v)]+ ['WVHT3','WVHT_q75_3']
        drop_rain_totals3 = [v for v in df.columns if 'lograin3_' in v]
        drop_list += drop_lag3  + drop_rain_totals3
    
    df = df[[c for c in df if c not in drop_list]]
    
    return df, drop_list

def fit_lm(X, y, score_metric, output_bin=True, seed=0, select_vars='all', multi=False, cv=5):  
    '''Fits Regression model after selecting variables
    X - calibration independent data; 
    y - calibration dependant variable;
    output_bin - is y a binary output? (or continuous)
    C - model regularization coefficient 
        - float: smaller - more regularization of variables);
        - If integer, test grid of size C to find optimal regulatization
    select_vars - variable selection method
        - ' all' - use all variables in X_train
        - 'force' - use only keep_vars to model with
        - else, perform feature selection
    cv - number of cross-validation steps
    scorer - model evaluation metric (roc_auc, accuracy)
    
    '''
    
    if output_bin:
        #score_metric='f1' # accuracy, f1, recall
        reg_param = 'C'
        lm = LogisticRegression(C = 1, 
                                penalty = 'elasticnet', # l1, l2, elasticnet, None 
                                l1_ratio = 0.5,
                                class_weight = 'balanced', # None, balanced
                                solver = 'saga',
                                random_state=seed)
        
    else:
        reg_param = 'alpha'
        #scorer='neg_root_mean_squared_error'  # r2, 
        lm = ElasticNet(alpha = 0.1, 
                        l1_ratio = 0.5,
                        random_state=seed)

    if select_vars in ['all','force']:
        features = X.columns
        
    else:
        ## Permutation method (narrow down big variable amounts)
        print('\nNarrowing down variables w/ permutation importances (scoring: ' + score_metric + ')')
        lm.fit(X,y)
        temp = permutation_importance(lm, X, y, 
                                      scoring=score_metric, 
                                      random_state=seed,
                                      n_repeats=cv)['importances_mean']
        temp = pd.Series(data=temp, index=X.columns).sort_values(ascending=False)
        #features = list(temp.index[0:10])
        features = list(temp[temp > 1.5*temp.mean()].index)  # Select the variables > X times the mean importance
        assert len(features) > 0, 'Linear Regression failed to select any variables'
        if len(features) < 3:
            features = list(temp.index[0:3])
        
        print('  ' + str(len(features)) + ' features selected')
        print(*features)
        X = X[features]
        

        c=0
        print('\nRFECV Variable Selection (scoring: ' + score_metric + ')')
        while c<1:
        
            ## Recursive Feature Elimination - Stepwise variable selection
            S = RFECV(lm, 
                      cv=cv, 
                      scoring=score_metric,
                      min_features_to_select=3,
                      verbose=0).fit(np.array(X), np.array(y))
            
            if multi:  # Check multicolinearity 
                old_len = len(features)    
                X = multicollinearity_check(X[features], thr=5)
                features = X.columns
                if len(features)<old_len:
                    c-=1
            c+=1   
            
        features = X.columns[list(np.where(S.support_)[0])]
        print('  ' + str(len(features)) + ' feature(s) selected')
        print(*features)
        
    ### Fit Model
    
    ## Grid Search for best parameters
    print('\nGrid Search for best model parameters (scoring: ' + score_metric + ')')
    gs = GridSearchCV(lm, 
                          param_grid={reg_param:[0.0001, 0.001, 0.01, 0.1, 1, 10],
                                      'l1_ratio': [0, 0.1, 0.25, .5, .75, 0.9, 1]},
                          cv=cv, 
                          scoring = score_metric,
                          verbose = 1)
    
    gs.fit(X[features], y)
    lm = gs.best_estimator_
    print('\n')
    print(lm)
    
    #lm.fit(X[features], y) # Fit model
    
    return list(features), lm


def fit_rf(X, y, score_metric, output_bin=True, n_trees=300, max_depth=3, max_features=.75, max_samples=.75, min_samples_leaf=3, seed=0, select_vars='all', cv=5):  
    '''Fits Random Forest model
    
    X - calibration independent data; 
    y - calibration dependant variable;
    select_vars - variable selection method
        - ' all' - use all variables in X_train
        - 'force' - use only keep_vars to model with
        - else, perform feature selection
    '''
    if output_bin:
        rf = RandomForestClassifier(n_estimators=n_trees, 
                                    oob_score=True,
                                    max_depth=max_depth,  # None - expands until all leaves are pure
                                    max_features=max_features,
                                    max_samples=max_features,
                                    min_samples_leaf=min_samples_leaf,
                                    class_weight='balanced', # None, 'balanced'
                                    random_state=seed,
                                    n_jobs=-1)
        #score_metric = 'recall' # accuracy, recall, f1, roc_auc
        
    else:
        rf = RandomForestRegressor(n_estimators=n_trees, 
                                   oob_score=True,
                                   max_depth=max_depth,  # None - expands until all leaves are pure
                                   max_features=max_features,
                                   max_samples=max_features,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=seed)
        
        #score_metric = 'neg_root_mean_squared_error' # r2, max_error, neg_mean_absolute_error, neg_root_mean_squared_error
        
    if select_vars in ['all','force']:
        features = X.columns
        
    else:  # Use variable selection method
    
        ## Random Forest Permutation method (narrow down big variable amounts)
        print('\nNarrowing down variables w/ permutation importances (scoring: ' + score_metric + ')')
        rf.fit(X,y)
        temp = permutation_importance(rf, X, y, 
                                      scoring=score_metric, 
                                      random_state=seed,
                                      n_repeats=cv)['importances_mean']
        temp = pd.Series(data=temp, index=X.columns).sort_values(ascending=False)
        #features = list(temp.index[0:10])
        features = list(temp[temp > 1.5*temp.mean()].index)  # Select the variables > X times the mean importance
        #assert len(features) > 0, 'Random Forest Regression failed to select any variables'
        if len(features) < 3:
            features = list(temp.index[0:10])
        
        print('  ' + str(len(features)) + ' features selected')
        print(*features)
        X = X[features]
        
        ## Recursive Feature Elimination - Stepwise variable selection
        print('\nRFECV Variable Selection (scoring: ' + score_metric + ')')
        S = RFECV(rf, 
                  cv=cv, 
                  scoring=score_metric,
                  min_features_to_select=3,
                  verbose=0).fit(np.array(X), np.array(y))
        
        features = X.columns[list(np.where(S.support_)[0])]
        print('\n' + str(len(features)) + ' feature(s) selected')
        print(*features)
        
    ### Fit Model
    
    ## Grid Search for best parameters
    print('\nGrid Search for best model parameters (scoring: ' + score_metric + ')')
    gs = GridSearchCV(rf, 
                          param_grid={'max_depth':[3,5],
                                      'max_features': [0.25, .5],
                                      #'max_samples': [.5,.75],
                                      'min_samples_leaf':[1, 3, 5]},
                          cv=cv, 
                          scoring = score_metric,
                          verbose = 1)
    
    gs.fit(X[features], y)
    rf = gs.best_estimator_
    print('\n')
    print(rf)
    
    # rf.fit(X[features], y) # Fit Model (simple)

    return list(features), rf

def fit_svm(X, y,score_metric, output_bin=True, C=0.01, kernel='linear', select_vars='all', cv=5, seed=0):  
    '''Fits Support Vector Machine model
    
    X - calibration independent data; 
    y - calibration dependant variable;
    select_vars - variable selection method
        - ' all' - use all variables in X_train
        - 'force' - use only keep_vars to model with
        - else, perform feature selection
    '''
    if output_bin:
        svm = SVC(C=C,
                  kernel=kernel, # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
                  max_iter = -1,
                  tol = 0.001,
                  class_weight='balanced', # None, 'balanced'
                  probability=True,
                  random_state=seed)
        
        #score_metric = 'f1' # accuracy, recall, f1, roc_auc, precision
        
    else:
        svm = SVR(kernel=kernel,
                  C=C)
        
        #score_metric = 'r2' # r2, max_error, neg_mean_absolute_error, neg_root_mean_squared_error
        
    if select_vars in ['all','force']:
        features = X.columns
        
    else:  # Use variable selection method
        svm.fit(X,y)
        
        # Random Forest Permutation method (narrow down big variable amounts)
        print('\nNarrowing down variables w/ permutation importances (scoring: ' + score_metric + ')')
        
        temp = permutation_importance(svm, X, y, 
                                      scoring=score_metric, 
                                      random_state=seed,
                                      n_repeats=cv)['importances_mean']
        temp = pd.Series(data=temp, index=X.columns).sort_values(ascending=False)
        #features = list(temp.index[0:10])
        features = list(temp[temp > 1.5*temp.mean()].index)  # Select the variables > X times the mean importance
        #assert len(features) > 0, 'SVM failed to select any variables'
        if len(features) < 3:
            features = list(temp.index[0:10])
        print('  ' + str(len(features)) + ' features selected')
        print(*features)
        X = X[features]
        
        ## Recursive Feature Elimination - Stepwise variable selection
        print('\nRFECV Variable Selection (scoring: ' + score_metric + ')')
        S = RFECV(svm, 
                  cv=cv, 
                  scoring=score_metric,
                  min_features_to_select=3,
                  verbose=0).fit(np.array(X), np.array(y))
        
        features = X.columns[list(np.where(S.support_)[0])]
        print('\n' + str(len(features)) + ' feature(s) selected')
        print(*features)
        
    ### Fit Model
    
    ## Grid Search for best parameters
    print('\nGrid Search for best model parameters (scoring: ' + score_metric + ')')
    gs = GridSearchCV(svm, 
                      param_grid={
                          'C':[0.001, 0.01, .1, 1, 10],
                          'kernel': ['rbf','poly','sigmoid'],
                          'gamma': ['auto', 'scale'],
                                  },
                      cv=cv, 
                      scoring = score_metric,
                      verbose = 1)
    
    gs.fit(X[features], y)
    svm = gs.best_estimator_
    print('\n')
    print(svm)
    print('gamma: ' + str(svm.gamma))
    
    # svm.fit(X[features], y) # Fit Model (simple)
    
    return list(features), svm
    
def fit_gbm(X, y, score_metric, output_bin=True, seed=0, select_vars='all', cv=5):  
    '''Fits Gradient Boosted Machine model
    
    X - calibration independent data; 
    y - calibration dependant variable;
    select_vars - variable selection method
        - ' all' - use all variables in X_train
        - 'force' - use only keep_vars to model with
        - else, perform feature selection
    '''
    if output_bin:
        gbm = GradientBoostingClassifier(#loss = 'exponential',
                                         n_estimators=100,
                                         learning_rate=0.1,
                                         subsample = 0.75, # .5, .75, 1
                                         #min_samples_split = 2, # 2
                                         min_samples_leaf=1,
                                         max_depth=3,  # None - expands until all leaves are pure
                                         max_features=0.75,
                                         n_iter_no_change = 10,  # 5, 10, None
                                         random_state=0)
        # score_metric = 'f1' # accuracy, recall, f1, roc_auc
        
    else:
        gbm = GradientBoostingRegressor(loss = 'squared_error',
                                         n_estimators=100,
                                         learning_rate=0.1,
                                         subsample = 0.9, # .5, .75, 1
                                         #min_samples_split = 2, # 2
                                         min_samples_leaf=1,
                                         max_depth=3,  # None - expands until all leaves are pure
                                         max_features=0.75,
                                         n_iter_no_change = 10,  # 5, 10, None
                                         random_state=0)
        
        #score_metric = 'neg_root_mean_squared_error' # r2, max_error, neg_mean_absolute_error, neg_root_mean_squared_error
        
    if select_vars in ['all','force']:
        features = X.columns
        
    else:  # Use variable selection method
    
        ## Random Forest Permutation method (narrow down big variable amounts)
        print('\nNarrowing down variables w/ permutation importances (scoring: ' + score_metric + ')')
        gbm.fit(X,y)
        temp = permutation_importance(gbm, X, y, 
                                      scoring=score_metric, 
                                      random_state=seed,
                                      n_repeats=cv)['importances_mean']
        temp = pd.Series(data=temp, index=X.columns).sort_values(ascending=False)
        #features = list(temp.index[0:10])
        features = list(temp[temp > 1.5*temp.mean()].index)  # Select the variables > X times the mean importance
        #assert len(features) > 0, 'Random Forest Regression failed to select any variables'
        if len(features) < 3:
            features = list(temp.index[0:10])
        
        print('  ' + str(len(features)) + ' features selected')
        print(*features)
        X = X[features]
        
        ## Recursive Feature Elimination - Stepwise variable selection
        print('\nRFECV Variable Selection (scoring: ' + score_metric + ')')
        S = RFECV(gbm, 
                  cv=cv, 
                  scoring=score_metric,
                  min_features_to_select=3,
                  verbose=0).fit(np.array(X), np.array(y))
        
        features = X.columns[list(np.where(S.support_)[0])]
        print('\n' + str(len(features)) + ' feature(s) selected')
        print(*features)
        
    ### Fit Model
    
    ## Grid Search for best parameters
    print('\nGrid Search for best model parameters (scoring: ' + score_metric + ')')
    gs = GridSearchCV(gbm, 
                      param_grid={'n_estimators': [100,200,300],
                                  'max_depth':[3,5,7],
                                  #'max_features': [.5,.75],
                                  'learning_rate': [0.01, 0.1,0.3],
                                  'subsample': [0.5, .75, 1],
                                  },
                      cv=cv, 
                      scoring = score_metric,
                      verbose = 1)
    
    gs.fit(X[features], y)
    gbm = gs.best_estimator_
    print('\n')
    print(gbm)
    
    # gbm.fit(X[features], y) # Fit Model (simple)

    return list(features), gbm
    
   
   
#%% Load and Process Data
#print('\n- - ' + beach + ' - -')
def load_data(beach, fib, train_years, test_years, save=True):
    beach_folder = os.path.join(base_folder, 'beach', beach.replace(' ', '_'))
    var_file = os.path.join(beach_folder, 'variables', beach.replace(' ', '_') + '__variables.csv')
    
    df = pd.read_csv(var_file, index_col=['date'], parse_dates=['date'])
    df = df[df.index.year.isin(list(range(min(train_years), max(test_years)+1)))]
    
    ### Set aside persistence data
    if output_bin:
        dep_var = fib + '_exc'
        persist = df[[
            fib + '_ant_exc', fib + '_ant1_exc',
            fib + '_ant2_exc', fib + '_ant3_exc'
            ]].copy()
    else:
        ep_var = 'log' + fib
        persist = df[[
            'log' + fib + '_ant', 'log' + fib + '_ant1',
            'log' + fib + '_ant2', 'log' + fib + '_ant3'
            ]].copy()
    
    ### Drop initial variables - irrelevant, 0 variance, other FIB
    drop_list = []
    
    ## Non-Applicable Variables
    # Varies dependant on FIB, lead_time
    
    ## Default
    drop_default = [v for v in df.columns if any(x in v for x in ['sample_time', 'MWD','wdensity','rad0','atemp0','upwell'])] + \
                 [v for v in df.columns if any(x in v for x in ['rain','flow','chl','turb']) and (('log' not in v) and ('q75' not in v))] 
    drop_list += drop_default
     
    ## Drop other FIB variables
    # We do not consider other FIB in the models
    other_fib = [f for f in fib_thresh if f != fib]
    drop_other_fib = []
    for f in other_fib:
        drop_other_fib += [v for v in df.columns if f in v]
    drop_list += drop_other_fib
        
    ## Drop same FIB variables that do not apply
    drop_same_fib = [v for v in df.columns if (fib in v) and (v != dep_var) and (v not in keep_vars)] # Drop non-transformed fib
    drop_list += drop_same_fib
    # varies depending on lead_time
    
    ## Drop Same Day variables
    drop_same_day = [v for v in df.columns if (not any(char.isdigit() for char in v)) \
                  and not any(x in v for x in ['tide','doy','dow','month','weekend']) \
                      and (fib not in v) ] + ['wspd_q75','WVHT_q75','DPD_q75', 'chl_q75','turb_q75']
    drop_list += [v for v in drop_same_day if v not in drop_list]
        
    ## Drop
    # remove from drop_list if in keep_vars
    drop_list = [d for d in drop_list if d not in keep_vars]
    
    print('\nDropped ' + str(len(np.unique(drop_list))) + ' irrelevant features')
    df = df[[c for c in df if c not in drop_list]]
    
    ## Remove 0 variance variables
    vt = VarianceThreshold().fit(df)
    if (vt.variances_ == 0).sum() > 0:
        print('Dropped ' + str((vt.variances_ == 0).sum()) + ' zero variance features:')
        print(list(df.columns[(vt.variances_ == 0)]))
    df = pd.DataFrame(data= vt.transform(df), 
                      index=df.index, 
                      columns = df.columns[vt.get_support(indices=True)])
    
    print('  ' + str(len(df.columns)) + ' vars remaining')


    ### Highly correlated variables
    ## remove highly correlated variables (Check VIF later)
    if check_correl:
        drop_corr = check_corr(df[dep_var], df[[v for v in df.columns if v not in [dep_var] + drop_list]])
        df = df[[c for c in df if c not in drop_corr]]
        print('  ' + str(len(df.columns)) + ' vars remaining')


    ### Deal w Missing Data 
    df = df.dropna(subset=[dep_var])  # drop all rows where FIB == NaN
    
    ## Drop columns with high missing %
    miss_frac = df.isnull().sum() / len(df)
    drop_missing = miss_frac[(miss_frac > miss_allow)].index
    drop_missing = [c for c in drop_missing if (c not in keep_vars) and (c != dep_var)]
    
    df = df.drop(drop_missing, axis=1)
    print('\nDropped ' + str(len(drop_missing)) + ' features with data > ' + str(100*miss_allow) + '% missing')
    
    ## Drop rows (days) with missingness in vars in keep_vars
    len_temp = len(df)
    df = df.dropna(axis=0)
    print('Dropped ' + str(len_temp - len(df)) + ' rows; ' + str(len(df)) + ' rows remaining')
    print(str(len(df.columns)) + ' variables remaining:')
    print(*df.columns, sep=', ')

    ### Partition Data into Train and Test Sets
    
    ### Train/Test Sets
    train_data = df[df.index.year.isin(train_years)].copy().sort_index()
    test_data = df[df.index.year.isin(test_years)].copy().sort_index()
    
    y_train = train_data[dep_var].copy()
    y_train.dropna(inplace=True)
    X_train = train_data.copy().drop(dep_var, axis=1)
    X_train = X_train.reindex(y_train.index)
    
    y_test = test_data[dep_var].copy()
    y_test.dropna(inplace=True)
    X_test = test_data.copy().drop(dep_var, axis=1)
    X_test = X_test.reindex(y_test.index)
    
    ### Stats and Save
    if output_bin:
        train_exc = y_train.sum()
        test_exc = y_test.sum()
    else:
        train_exc = (y_train > np.log10(fib_thresh[fib]+1)).sum()
        test_exc = (y_test > np.log10(fib_thresh[fib]+1)).sum()
        
    print('\nTrain (' + str(min(train_data.index.year)) + '-' + str(max(train_data.index.year)) + '):')
    print('  Samples - ' + str(len(y_train)) + '\n  Exc. - ' + str(train_exc))
    print('Test (' + str(min(test_data.index.year)) + '-' + str(max(test_data.index.year)) + '):')
    print('  Samples - ' + str(len(y_test)) + '\n  Exc. - ' + str(test_exc))
    
    # Save test and training datasets to seperate CSV files
    if save:
        ### Create FIB and partition subfolders
        partition_folder = os.path.join(base_folder, 'beach', beach, 'models', 
                                        fib, 'train_' + str(min(train_years))[-2:] + \
                                            '_' + str(max(train_years))[-2:] + \
                                            '_test_' + str(min(test_years))[-2:] + \
                                            '_' + str(max(test_years))[-2:])
        os.makedirs(partition_folder, exist_ok=True)
        
        ### Save Train and Test datasets
        train_data.to_csv(os.path.join(partition_folder, 'train_data.csv'))
        test_data.to_csv(os.path.join(partition_folder, 'test_data.csv'))
        
    return [y_train, X_train, y_test, X_test, persist, drop_list, drop_missing, partition_folder]
    
#%% Fit and Evaluate Models
def model(y_train, X_train, y_test, X_test, fib, lead_time, model_type, partition_folder, save=True):
    ''' Fit, evaluate, and save predictive model for given FIB and lead time'''
    
    if output_bin:
        exc_thresh = 0.5
    else:
        exc_thresh = np.log10(fib_thresh[fib] + 1)  
    
    ### Persistence
    if model_type == 'per':
        
        if lead_time > 0:
            pvar = [c for c in X_train.columns if str(lead_time) in c][0]
        else: # nowcast
            pvar = [c for c in X_train.columns if not any(map(str.isdigit, c))][0]
        
        persist_train = X_train[pvar].reindex(y_train.index)
        persist_test = X_test[pvar].reindex(y_test.index)
        
        ## Performance
        print('\n* * Persistence (Current) Method * *')
        print(y_test.name + ' = ' + pvar)
        print('\nTrain:')
        train_perf = model_eval(y_train, persist_train, thresh=exc_thresh, output_bin=output_bin)
        print(train_perf)
        print('Test:')
        test_perf = model_eval(y_test, persist_test, thresh=exc_thresh, output_bin=output_bin)
        print(test_perf)
        print('\n')
    
    
    ### Predictive Models
    else:
        ### Drop lead time vars
        #print('\nDropping variables not applicable to lead time ' + str(lead_time))
        X_train, drop_lag = drop_lead_time_vars(X_train, lead_time)
        X_test, _ = drop_lead_time_vars(X_test, lead_time)
        #print('  Removed ' + str(len(np.unique(drop_lag))) + ' variables')
        
        ### Scale Inputs
        if vs_method == 'force':  # adjust to only keep_vars
            X_train = X_train[[k for k in keep_vars if k in X_train.columns]]
            X_test = X_test[[k for k in keep_vars if k in X_test.columns]]
            print('\n\nForcing models with ' + str(len(keep_vars)) +' variables:')
            print(keep_vars)
        
        if scale:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = pd.DataFrame(data=scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
            X_test = pd.DataFrame(data=scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
         
        ### Regression
        if model_type == 'lm':
            print('\n* * Regression  * * ')
            features, model = fit_lm(X_train, y_train, score_metric, output_bin=output_bin, select_vars=vs_method)
            
            if output_bin:
                #print('\nBinary Logistic Regression:')
                print(str(round(model.intercept_[0], 2)) + ' - intercept')
                for f in range(0,len(model.coef_[0])):
                    print(str(round(model.coef_[0][f], 4)) + ' - ' + features[f])
                
            else:
                #print('\nMultiple Linear Regression:')
                print(str(round(model.intercept_, 2)) + ' - intercept')
                for f in range(0,len(model.coef_)):
                    print(str(round(model.coef_[f], 2)) + ' - ' + features[f])
                #print('\nSelected alpha: ' + str(model.alpha_))
        
        ### Random Forest
        elif model_type == 'rf':
            print('\n* * Random Forest * *')
            features, model = fit_rf(X_train, y_train,score_metric, output_bin=output_bin, select_vars=vs_method)
            
            print('\nFeature Importances: ')
            for f in range(0,len(features)):
                print(str(round(model.feature_importances_[f], 2)) + ' - ' + features[f])
        
        ### Support Vector Machine
        elif model_type == 'svm':
            print('\n* * Support Vector Machine * *')
            features, model = fit_svm(X_train, y_train,score_metric, output_bin=output_bin, select_vars=vs_method)
        
        ### Gradient Boosted Machine
        elif model_type == 'gbm':
            print('\n* * Gradient Boosted Machine * *')
            features, model = fit_gbm(X_train, y_train,score_metric, output_bin=output_bin, select_vars=vs_method)
            
            print('\nFeature Importances: ')
            for f in range(0,len(features)):
                print(str(round(model.feature_importances_[f], 2)) + ' - ' + features[f])
            
        
        ### Evaluate on Training 
        #y_train_pred = model.predict(X_train[features])
        y_train_pred = model.predict_proba(X_train[features])[:,1]
        print('\nTraining: ')
        train_perf = model_eval(y_train, y_train_pred, thresh=exc_thresh, output_bin=output_bin)
        print(train_perf)
        
        
        ### Evaluate on Test
        #y_test_pred = model.predict(X_test[features])
        y_test_pred = model.predict_proba(X_test[features])[:,1]
        print('Testing: ')
        test_perf = model_eval(y_test, y_test_pred, thresh=exc_thresh, output_bin=output_bin)
        print(test_perf)
        print('\n')
            
    

    ### Save files
    if save:
        ### Create subfolders
        model_folder = os.path.join(partition_folder, 'lt_' + str(lead_time), model_type)
        os.makedirs(model_folder, exist_ok=True)
        
        ### Model Variables / PKL
        if model_type != 'per':
            
            pd.Series(features, name='features').to_csv(os.path.join(model_folder, 
                                                                     model_type + '_features.csv'), index=False)
          
            model_file = model_type + '_model' + '.pkl'
            joblib.dump(model, os.path.join(model_folder, model_file))
            # use joblib.load to load this file in the model runs script
        
        ### Performance
        df_perf = pd.DataFrame(data = [train_perf,test_perf], index = ['train','test'])
        df_perf.to_csv(os.path.join(model_folder, model_type + '_performance.csv')) 
    
    