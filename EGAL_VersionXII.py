#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:10:40 2019

@author: billcoleman
"""

import munging_functions
import agg_Clustering
import egal_functions
import trainTest_functions
import random_usc

#import os
import random

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing # for normalisation
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt

from sklearn.svm import SVC # SVM model
from sklearn.preprocessing import MinMaxScaler # to normalise data
from sklearn.metrics import make_scorer, recall_score, precision_score, \
accuracy_score, balanced_accuracy_score # line return here - problem??
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from numpy import array


'''
############################################

Also, remember wrt axis:
From: https://stackoverflow.com/questions/22149584/what-does-axis-in-pandas-mean/22149930

Axis 0 will act on all the ROWS in each COLUMN
Axis 1 will act on all the COLUMNS in each ROW

So a mean on axis 0 will be the mean of all the rows in each column, and a mean 
on axis 1 will be a mean of all the columns in each row.
'''




'''
############################################
########### GLOBAL VARIABLES ###############
############################################

In order to access these global variables in a local scope, remember to reference 
them in the local scope by declaring...

global variableName

... somewhere in the local scope

'''

# boolean switch to control if we take an instance from this aggCluster
cluster_Empty = [False, False, False, False, False] 

# an index to train models from - labelled training pool
S1_index = [] #

# make an order to get instances from to make sure extra instances come from biggest aggCluster
cluster_order = [1, 2, 3, 0, 4]

# Array to hold accuracy values from train_Test()
trainAcc = []
testAcc = []
labelled_instances = [] # count no. labelled instances for more accurate plot
classReport = {}
confusion_Matrices = []
y_true_list = []
y_pred_list = []

# COMMENT THIS BACK IN IF STARTING FROM SCRATCH
cluster_means_dfs = {} # df of mean distances by aggCluster

U = [] # Index to Unlabelled set, we'll fill this with data from the model test set, 
       # essentially the instances that haven't been labelled yet
L = [] # Index to labelled set

count = 0 # global variables declared here so they can be accessed in multiple functions
cl_count = 0 # see: load_initial_instances() and get_instance_pwDist()

# to control ratio of diversity versus density
w = 0.5 # tried 0.25 no difference

# Number of Labels to get
NoL = 50 # was using 10 initially, changed it to speed up process

# set a max number of iterations and a boolean for stopping
max_iterations = 301
stop_all = False

# Reset beta_
# beta_ = alpha_ # comment back in if required for testing

# tracking EGAL Loops
loop_counter = 0
score_counter = 0

        
'''
############################################
########### INITIAL FINCTIONS ##############
############################################

ON INITIAL LOAD IF NO VARIABLES GENERATED

Run these commands \/\/\/ to generate variables initially:

'''

D_all, Cl_all = munging_functions.load_data()

D = munging_functions.make_D(D_all, Cl_all)

max_chroma, min_chroma = munging_functions.find_max_min()

chroma_X_scaled = munging_functions.flatten_chroma()

chroma_X_df = pd.DataFrame(chroma_X_scaled)

pairwiseDist = agg_Clustering.pairwise_Dist(chroma_X_df)

pairwiseDist, pwDist_mean, pwDist_std, pwDist_max, pwDist_min, alpha_, beta_ = \
agg_Clustering.pairwise_Dist(chroma_X_df)

# for controlling uper bound of instances elligible for CS
track_beta = []
track_beta.append(beta_)

# for controlling uper bound of instances elligible for CS
beta_old = pwDist_max

cl_Boolean_idx, D_by_cluster, chroma_X_by_cluster, cluster_pwDist = \
agg_Clustering.cluster_euclidean_distances(D, chroma_X_df)

cluster_means_dfs = agg_Clustering.aggCluster_mean_distances(cluster_pwDist)


# Now we need to select candidates for labelling on the basis of density/diversity
# First we calculate density values
# For each instance in U, this function sums all the eucl_distance values that
# are <= alpha_ so it looks at all the instances that are with 'range' of the
# instance and sums these - this is the density measure, a measure of 'clusterness'. 

# # Get initial centroids from aggCLusters
agg_Clustering.load_aggCluster_initInstances(NoL)
# 
# For storing test scores - comment out if compiling during testing
#testAcc_vault = []
#trainAcc_vault = []

# # Run initial train/test
trainTest_functions.train_Test()
# 
# # Get density_dict
denseVals_sort, density_dict, denseVals_sort_noZeros = egal_functions.get_sorted_density_values()
# =============================================================================
#
# # Next calculate diversity values, and use these to filter U to instances
# # which are outside the radius of beta_, which we then sort so we find the most
# # dense instances
# 
# EGAL LOOP
#
# egal_functions.egal_loop()
# egal_functions.egal_loopII()
# random_usc.random_loop()
# random_usc.usc_loop()
# =============================================================================


def plotTrainTest():
    
    test = testAcc_vault[-1]
    train = trainAcc_vault[-1]
    
    # Plot training & validation accuracy values
    plt.plot(test)
    plt.plot(train)
    plt.plot([0, len(test)-1], [test[0], test[-1]], '--')
    plt.title('Train and Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Data Capture every 10th EGAL Loop')
    plt.legend(['Test', 'Train'], loc='lower right')
    #plt.savefig('D1-2-3_trValAcc.png')
    plt.show()

def store_test_runs():
    
    global testAcc_vault
    global trainAcc_vault
    
    testAcc_vault.append(testAcc)
    trainAcc_vault.append(trainAcc)





'''
############################################
############ HANDY SNIPPETS ################
############################################



Sort a dataframe by a column
check = check.sort_values(by=['dense'], ascending=False)

Filter a dataframe by values in a column checked against a variable
check = check.loc[check['diverse'] > beta_]

convert values of a dict to a list
dense_Vals = list(density_dict.values())

Make a dict out of separate lists/arrays
make_dict = {'dense':dense_Vals, 'U':U, 'diverse':div}

Make a dataframe out of a dict
_df_dense_idx_diverse = pd.DataFrame(make_dict)

In a classification vector such as y_train, gives a count of numbers in each class
np.bincount(y_train)
Out[694]: array([18,  2])

'''