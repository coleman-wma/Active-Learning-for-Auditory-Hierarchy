#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:32:24 2019

@author: billcoleman
"""

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC # SVM model
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict

from sklearn.metrics import make_scorer, recall_score, precision_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from sklearn.utils.multiclass import type_of_target

from datetime import datetime

# =============================================================================
# with open('data/FEATURES_JULY_mfcc_chroma_lpms.data', 'rb') as the_data:  
#         # read the data as binary data stream
#         all_featureSets = pickle.load(the_data)
# =============================================================================

'''
Split the data into 5 outer stratified folds
'''

# Stratified k Folds - outer split of 5
skf_Outer = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)

# dicts to hold outer splits and indexes
ca_out_trainval_indices = {} 
ca_out_test_indices = {}

dict_no = 0

# Stepping through the Outer folds one at a time
# D['Cat'] is a series of floats that needs to be cast as ints to be binary
for trainval_index, test_index in skf_Outer.split(all_featureSets[0],
                                                  D['Cat'].astype(int)):
    
    # keep track of different indices for later
    # An index to c. 2401 train/validation instances
    ca_out_trainval_indices[dict_no] = trainval_index
    # The corresponding index to test instances - c.600 of them
    ca_out_test_indices[dict_no] = test_index
    
    dict_no += 1

'''
For each trainval index above I apply stratified CV to it and fit models for
each, testing them on the corresponding test index above.

Do this for each dataset and see is there a big difference in performance.

Best datasets should be selected for AL piece.
'''

# Stratified k Folds - inner split of 4
skf_Inner = StratifiedKFold(n_splits=4, random_state=3, shuffle=True)

ca_inner_data = all_featureSets[0].iloc[ca_out_trainval_indices[0]]
ca_inner_labels = D['Cat'].iloc[ca_out_trainval_indices[0]].astype(int)

# On each inner split, train model, get balanced accuracy score
for train_index, val_index in skf_Inner.split(ca_inner_data, ca_inner_labels):
    
    print(train_index.shape, val_index.shape)
    

# Number of random trials
NUM_TRIALS = 1

labels = D['Cat'].astype(int)

# Set up possible values of parameters to optimize over
p_grid = {"C": [1],  # "C": [1, 10, 100],
          "gamma": [0.1]}  # "gamma": [.01, .1]}

# Set up possible values of parameters to optimize over
# p_grid_l = {"C": [1]}  # "gamma": [.01, .1]}
p_grid_l = {"C": [0.001, 0.10, 0.1, 1, 10, 25, 50, 100, 1000, 10000]}

# c_values = [0.001, 0.10, 0.1, 1, 10, 25, 50, 100, 1000, 10000]
c_values = [1]

# kernels to run
kernels = ['rbf', 'linear', 'poly', 'sigmoid']

# Array to store scores
# balanced_acc = []
# conf_mats = []

# Loop for each dataset
# datasets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# datasets = [16, 21, 22]
# datasets = [25, 26, 27, 28, 29]
# datasets = [0, 8, 9, 30]
datasets = [26]

'''
For running multiple kernels
'''

for k in datasets:
    data = all_featureSets[k]
    print("#############################################")
    print("######## Starting Dataset: ", k, "###########")
    print("#############################################")

    # Loop for each kernel
    for j in kernels:
        
        print("Starting kernel: ", j)
        
        # model to run
        svm = SVC(kernel = j, gamma="scale", class_weight='balanced')
        
        # Loop for each trial
        for i in range(NUM_TRIALS):
        
            # Choose cross-validation techniques for the inner and outer loops,
            # independently of the dataset.
            # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
            inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
            
            for c in c_values:
                grid = {"C": [c]}
                print("Starting C Value: ", c)
                start = time.time()
            
                # Non_nested parameter search and scoring
                clf = GridSearchCV(estimator=svm, param_grid=grid, cv=inner_cv,
                                   iid=False)
            
                # Nested CV with parameter optimization
                nested_score = cross_val_score(clf,
                                               X=data,
                                               y=labels,
                                               cv=outer_cv,
                                               scoring='balanced_accuracy')
                balanced_acc.append(nested_score.mean())
                
                y_pred = cross_val_predict(clf,
                                           X=data,
                                           y=labels,
                                           cv=outer_cv)
                
                conf_mat = confusion_matrix(labels, y_pred)
                conf_mats.append(conf_mat)
                
                class_report_ = classification_report(labels, y_pred)
                
                elapsed_time = time.time() - start
                
                print("C Value: ", c)
                print("Dataset: ", k)
                print("FINISHED Kernel: ", j)
                print("Time taken in minutes: ", elapsed_time * 0.01666)
                print("###################################################")
                print("Average Balanced Accuracy across 5 Folds: ", balanced_acc[-1])
                print("###################################################")
                print("Confusion matrix:\n{}".format(conf_mat))
                print("###################################################")
                print("Classification Report ", class_report_)
                print("###################################################")

'''
For fitting parameters on one kernel
'''
# Set labels
labels = D['Cat'].astype(int)

# Set up possible values of parameters to optimize over
p_grid = {'gamma': ['scale', 'auto', 10, 1, 0.1, 0.01, 0.001, 0.0001],
          'C': [1, 10]}

scores = ['balanced_accuracy'] # , 'accuracy_score']

# Choose cross-validation techniques for the inner and outer loops,
# independently of the dataset.
# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

datasets = [9]


for k in datasets:
    
    print("#############################################")
    print("######## Starting Dataset: ", k, "###########")
    print("#############################################")

    for split in range(len(ca_out_trainval_indices)):
        
        # set dataset
        data = all_featureSets[k].iloc[ca_out_trainval_indices[split]]
        labs = labels.iloc[ca_out_trainval_indices[split]]
        
        print("#############################################")
        print("######## Starting Split: ", split, "#########")
        print("#############################################")
            
        # model to run
        svm = SVC(kernel='rbf', class_weight='balanced')
        
        # Loop for each trial
        for score in scores:
    
            print("# Tuning hyper-parameters for %s" % score)
            print()
            
            start = time.time()
        
            # Non_nested parameter search and scoring
            clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,
                               iid=False, scoring=score)
            
            clf.fit(data, labs)
    
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
        
# =============================================================================
#             # Nested CV with parameter optimization
#             nested_score = cross_val_score(clf,
#                                            X=data,
#                                            y=labels,
#                                            cv=outer_cv,
#                                            scoring='balanced_accuracy')
#         
#             y_pred = cross_val_predict(clf,
#                                        X=data,
#                                        y=labels,
#                                        cv=outer_cv)
#             
#             conf_mat = confusion_matrix(labels, y_pred)
#             
#             class_report_ = classification_report(labels, y_pred)
# =============================================================================
            
            elapsed_time = time.time() - start
            
            print("Dataset: ", k)
            print("Split: ", split)
            print("Time taken in minutes: ", elapsed_time * 0.01666)
            print("###################################################")


for i in range(len(ca_out_test_indices)):
    
    print("Starting Fold: ", i)
    
    dateTimeObj = datetime.now()
    print(dateTimeObj.hour, ':', dateTimeObj.minute, ':', dateTimeObj.second)
    
    # train data from ca_out_trainval_indices
    X_train = all_featureSets[9].iloc[ca_out_trainval_indices[i]]
    # test data from ca_out_test_indices
    X_test = all_featureSets[9].iloc[ca_out_test_indices[i]]
    # train labels from D index S1
    y_train = D['Cat'].iloc[ca_out_trainval_indices[i]]
    # test labels from D index everything
    y_test = D['Cat'].iloc[ca_out_test_indices[i]]
    
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    
    # Parameters fit in CV process
    clf = SVC(kernel='rbf',
              C=1,
              gamma=0.01,
              class_weight='balanced')
    
    # Fit model
    clf.fit(X_train, y_train) # .values.ravel()
    
    # BALANCED accuracy scores
    # y_pred_dec = clf.decision_function(X_test)
    y_true, y_pred = y_test, clf.predict(X_test)        
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print("------------------------------------------------------------------")
    print('Balanced accuracy on test set (y_true Vs y_pred): ', bal_acc)
    print("------------------------------------------------------------------")
    
    conf_mat = confusion_matrix(y_true, y_pred)
    
    class_report_ = classification_report(y_true, y_pred)
    
    print("Finished Fold: ", i)
    dateTimeObj = datetime.now()
    print(dateTimeObj.hour, ':', dateTimeObj.minute, ':', dateTimeObj.second)

    print("###################################################")
    print("Balanced Accuracy: ", bal_acc)
    print("###################################################")
    print("Confusion matrix:\n{}".format(conf_mat))
    print("###################################################")
    print("Classification Report ", class_report_)
    print("###################################################")


'''
For fitting parameters on random splits 
'''

# Set labels
labels = D['Cat'].astype(int)

# Set up possible values of parameters to optimize over
p_grid = {'gamma': ['scale', 'auto', 10, 1, 0.1, 0.01, 0.001, 0.0001],
          'C': [1, 10]}

scores = ['balanced_accuracy'] # , 'accuracy_score']

# Choose cross-validation techniques for the inner and outer loops,
# independently of the dataset.
# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=9)

datasets = [25]

data = all_featureSets[25]
labs = labels.iloc[all_featureSets[25].index]
    
# model to run
svm = SVC(kernel='rbf', class_weight='balanced')

# Loop for each trial
for score in scores:
    
    dateTimeObj = datetime.now()
    print("START TIME: ", dateTimeObj.hour, ':', dateTimeObj.minute, ':', dateTimeObj.second)

    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    start = time.time()

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,
                       iid=False, scoring=score)
    
    clf.fit(data, labs)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    
    elapsed_time = time.time() - start
    
    dateTimeObj = datetime.now()
    print("FINISH TIME: ", dateTimeObj.hour, ':', dateTimeObj.minute, ':', dateTimeObj.second)
    print("###################################################")


for i in range(len(ca_out_test_indices)):
    
    print("Starting Fold: ", i)
    
    dateTimeObj = datetime.now()
    print(dateTimeObj.hour, ':', dateTimeObj.minute, ':', dateTimeObj.second)
    
    # train data from ca_out_trainval_indices
    X_train = all_featureSets[9].iloc[ca_out_trainval_indices[i]]
    # test data from ca_out_test_indices
    X_test = all_featureSets[9].iloc[ca_out_test_indices[i]]
    # train labels from D index S1
    y_train = D['Cat'].iloc[ca_out_trainval_indices[i]]
    # test labels from D index everything
    y_test = D['Cat'].iloc[ca_out_test_indices[i]]
    
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    
    # Parameters fit in CV process
    clf = SVC(kernel='rbf',
              C=1,
              gamma=0.01,
              class_weight='balanced')
    
    # Fit model
    clf.fit(X_train, y_train) # .values.ravel()
    
    # BALANCED accuracy scores
    # y_pred_dec = clf.decision_function(X_test)
    y_true, y_pred = y_test, clf.predict(X_test)        
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print("------------------------------------------------------------------")
    print('Balanced accuracy on test set (y_true Vs y_pred): ', bal_acc)
    print("------------------------------------------------------------------")
    
    conf_mat = confusion_matrix(y_true, y_pred)
    
    class_report_ = classification_report(y_true, y_pred)
    
    print("Finished Fold: ", i)
    dateTimeObj = datetime.now()
    print(dateTimeObj.hour, ':', dateTimeObj.minute, ':', dateTimeObj.second)

    print("###################################################")
    print("Balanced Accuracy: ", bal_acc)
    print("###################################################")
    print("Confusion matrix:\n{}".format(conf_mat))
    print("###################################################")
    print("Classification Report ", class_report_)
    print("###################################################")

