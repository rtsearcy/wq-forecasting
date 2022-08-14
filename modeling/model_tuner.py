#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_tuner.py

Takes in model and training data and tunes to find an optimal
probability decision threshold


Created on Mon Aug  8 17:43:28 2022

@author: rtsearcy
"""

def model_tuner(model, X, y, maximize='sens', min_sens=0.3, min_spec=0.8, scale=True): 
    
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve
    from beach_model import model_eval
    
    ## Scale Vars
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(data=scaler.transform(X), index=X.index, columns=X.columns)
    
    ## Maybe: Train/Val Split
    
    ## Probability predictions for each threshold
    y_pred = model.predict_proba(X)[:,1]
    
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    spec = np.round(1 - fpr, 3)
    sens = np.round(tpr, 3)
    
    df_tune = pd.DataFrame(data=[thresholds,sens,spec], index=['thresh','sens','spec']).T
        
    if maximize == 'sens':
        df_tune = df_tune[(df_tune.spec >= min_spec)]
        assert len(df_tune) > 0, 'No threshold available to tune spec >= ' + min_spec
        thresh = df_tune.iloc[-1]['thresh']
        
    elif maximize == 'spec':
        df_tune = df_tune[(df_tune.sens >= min_sens)]
        assert len(df_tune) > 0, 'No threshold available to tune sens >= ' + min_sens
        thresh = df_tune.iloc[0]['thresh']
    
    return thresh