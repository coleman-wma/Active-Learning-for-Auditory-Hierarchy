#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:08:13 2019

@author: bill
"""

import scipy.interpolate as interpolate
import numpy as np
import matplotlib.pyplot as plt
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html?highlight=friedman#scipy.stats.friedmanchisquare
from scipy.stats import wilcoxon, friedmanchisquare
import pickle
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import metrics
from scipy import stats
import pandas as pd
from numpy.random import seed
from numpy.random import randn
from sklearn import linear_model
import munging_functions

from sklearn.svm import SVC  # SVM model
from sklearn.metrics import balanced_accuracy_score


# =============================================================================
# D_all, Cl_all = munging_functions_imac.load_data()
# 
# D = munging_functions_imac.make_D(D_all, Cl_all)
# =============================================================================

'''
Demsar (2006) identifies Wilcoxon rank sum as being suitable for use between
two samples. The vectors you feed to the test must be of equal size however.

Metrics then will be:
    1) no_labels Vs bal_acc plots - compare methods
    2) Generate an AUC score for each curve - compare methods
    3) Statistical significance tests on the plotting point vectors,
    specifically I'm going to use Friedman for more than 3 samples and
    Wilcoxon rank sum as a post hoc test - remember to divide significance
    level by the number of samples if applying the Wilcoxon as a post for a
    Friedman test (Bonferroni correction)

'''


# =============================================================================
# # Load flattened datasets
# with open('data/FEATURES_JULY_mfcc_chroma_lpms.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     all_featureSets = pickle.load(filehandle)
# 
# for ft in all_featureSets:
#     all_featureSets[ft] = pd.DataFrame(all_featureSets[ft])
# =============================================================================


def get_auc(a):
    '''
    Get the auc score for a labelling run from just the run index
    '''

    auc = metrics.auc(labelled_inst_balAcc[a], balAcc_vault[a])

    return auc


def get_auc_provide_data(labels, accuracy_values):
    '''
    Get the auc score for a labelling run from provided data vectors
    '''

    auc = metrics.auc(labels, accuracy_values)

    return auc


'''
GETTING AN EXACT NUMBER OF LABELS THAT ACHIEVE 90% ACCURACY
'''

# declare a linear model instance
lin_model = linear_model.LinearRegression()

def get_accvalue_label(datarun, accvalue):
    '''
    I need a function that gets a number of labels value where balanced
    accuracy hits 90%.
    Takes a number which corresponds to a training run [datarun]
    value to predict number of labels for, in this case, 0.9 [accvalue]
    Find the plot points either side of 0.9 for both balAcc and labels
    fit linear model
    predict corresponding value for 0.9
    '''
    
    labels = labelled_inst_balAcc[datarun]
    acc = balAcc_vault[datarun]
    
    idx_un, idx_ov = find_nearest_acc(acc, accvalue)
    
    # Train the model using the available data
    lin_model.fit([[acc[idx_un]], [acc[idx_ov]]],
                  [[labels[idx_un]], [labels[idx_ov]]])
    # the point to predict for
    pt_pred = np.array([[accvalue]])
    # get the predicted value
    interp_point = round(float(lin_model.predict(pt_pred)))
    
    return interp_point


def find_nearest_acc(array, value):
    '''
    Finds the closest plotted points to a specified [value] in a supplied
    [array]. I can then fit a model to these two points to predict a label
    number for the [value] (in this instance, accuracy) I want to predict for.
    Used in get_accvalue_label()
    '''
    array_ = np.array(array)
    idx = (np.abs(array_-value)).argmin()

    if idx > value:
        idx_o = idx
        idx_u = idx - 1
    else:
        idx_u = idx
        idx_o = idx + 1

    # print("Value: ", value, "Under: ", array_[idx_u], "Over: ", array_[idx_o])
    return idx_u, idx_o

# =============================================================================
# # EGAL on featuresets
# plot_these = [179, 201, 202, 203, 204, 180, 181, 205, 206, 207, 209, 182, 183, 184]
# 
# # USAL on featuresets
# plot_these = [210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 221, 222, 223, 224]
# 
# plot_these = [252, 257, 254, 255]
# 
# for i in plot_these:
#     print(get_accvalue_label(i, 0.9))
# 
# # EGAL parameter search on best feature set - getting AULC values
# for i in range(225, 246):
#     print(round(get_auc(i), 2))
#     
# # EGAL parameter search on best feature set - getting labels for 90% acc
# for i in range(225, 246):
#     print(get_accvalue_label(i, 0.9))
# =============================================================================


def get_wilcoxon_tests():
    
    end_points = [5, 10, 20, 50, 251]
    
    for i in end_points:
        labs = labelled_inst_balAcc[61][i - 1]
        print(f"Random Versus EGAL w=0 for {labs} labels: ",
              wilcoxon(balAcc_vault[61][:i], balAcc_vault[62][:i]))
        print(f"Random Versus EGAL w=0.5 for {labs} labels: ",
              wilcoxon(balAcc_vault[61][:i], balAcc_vault[63][:i]))
        print(f"Random Versus EGAL w=1 for {labs} labels: ",
              wilcoxon(balAcc_vault[61][:i], balAcc_vault[64][:i]))
        print(f"Random Versus USAL for {labs} labels: ",
              wilcoxon(balAcc_vault[61][:i], balAcc_vault[70][:i]))
        print(f"EGAL w=0 Versus EGAL w=0.5 for {labs} labels: ",
              wilcoxon(balAcc_vault[62][:i], balAcc_vault[63][:i]))
        print(f"EGAL w=0 Versus EGAL w=1 for {labs} labels: ",
              wilcoxon(balAcc_vault[62][:i], balAcc_vault[64][:i]))
        print(f"EGAL w=0 Versus USAL for {labs} labels: ",
              wilcoxon(balAcc_vault[62][:i], balAcc_vault[70][:i]))
        print(f"EGAL w=0.5 Versus EGAL w=1 for {labs} labels: ",
              wilcoxon(balAcc_vault[63][:i], balAcc_vault[64][:i]))
        print(f"EGAL w=0.5 Versus USAL for {labs} labels: ",
              wilcoxon(balAcc_vault[63][:i], balAcc_vault[70][:i]))
        print(f"EGAL w=1 Versus USAL for {labs} labels: ",
              wilcoxon(balAcc_vault[64][:i], balAcc_vault[70][:i]))
