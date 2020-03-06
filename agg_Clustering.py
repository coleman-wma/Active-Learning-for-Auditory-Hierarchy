#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:07:34 2019

@author: billcoleman
"""

'''
####################################################
########### AGGLOMORATIVE CLUSTERING ###############
####################################################

# P4 pairwise similarity of all instances in D
# using euclidean distance because that's what I used in calculating the aggClusters
'''

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
import pandas as pd
import pickle

from sklearn.cluster import AgglomerativeClustering

from scipy.stats import iqr

_alpha_beta_etc_values = []

# =============================================================================
# with open('data/FEATURES_concat_feats.data', 'rb') as filehandle:  
#         # read the data as binary data stream
#         all_featureSets = pickle.load(filehandle)
# =============================================================================

# supply all_featureSets to this function
def aggCluster_forEach_featureSet(all_fSets):
    
    # just_these = [0, 4, 5, 6, 7, 8]
    #just_these = [0, 1, 2, 3]
    names = ['mfcc', 'mfcc_Del', 'mfcc_Del2', 'mfcc_Del5', 'chroma',
             'chroma_Del', 'chroma_Del2', 'chroma_Del5', 'lpms', 'lpms_scaled',
             'lpms_del', 'lpms_del_scaled', 'lpms_del2', 'lpms_del2_scaled',
             'lpms_del5', 'lpms_del5_scaled']
    ct = 0
    
    # a structure to hold cluster membership info per featureSet
    _cluster_repo = {}
    
    for featset in range(10, 16):  # all_fSets:
        
        print("featset is: ", featset)
        
        _cluster_repo[featset] = find_agg_cluster(all_fSets[names[featset]],
                     names[featset])
        
        print("featset = ", featset)
        print("Names = ", names[featset])
        
        ct += 1
        
    return _cluster_repo


# supply all_featureSets to this function
def aggCluster_forEach_featureSet_melspec(all_fSets):

    # a structure to hold cluster membership info per featureSet
    _cluster_repo = {}
        
    _cluster_repo[0] = find_agg_cluster(all_fSets)

    print("featset = ", 0)
    print("Count of names = ", 0)

    return _cluster_repo

        
def find_agg_cluster(data):  # , name):
    
    # affinity='cosine', linkage='single' was successful on some lpms data
    # 'euclidean*', 'manhattan', 'cosine' -- 'ward*', 'complete', 'average', 'single'
    cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
    clusterRes = cluster.fit_predict(data)
    
    as_series = pd.Series(clusterRes, index=data.index) # , columns=[name])
    
    return as_series
    # make clusterRes into a df to see


def pairwise_Dist(this_df, method): # of chroma_X_df
    pwD = euclidean_distances(this_df)  #(chroma_X, chroma_X)

    # pwD has a shape (3002, 3002) so it compares all instances to each other
    # and returns a grid of similarity values

    '''
    # P5 & P6 - calculate alpha and initial beta
    '''

    # calculate mean and st. deviation for the similarity matrix
    _mean, _std = pwD.mean(), pwD.std()
    _max, _min = pwD.max(), pwD.min()
    _var = np.var(pwD) # variance
    _iqr = iqr(pwD) # interquartile range
    
    if method == 0:
        _alpha = _beta = _mean - 0.5 * _std # 5.323
    elif method == 1:
        _alpha = _beta = _mean + 0.5 * _std # 7.391
    elif method == 2:
        _alpha = _beta = _mean - _std # 4.288
    elif method == 3:
        _alpha = _beta = _mean - _var # 2.079
    elif method == 4:
        _alpha = _beta = _mean + _var # 10.634
    
    _alpha_beta_etc_values.extend([_mean, _std, _var, _iqr, _max, _min, _alpha, _beta])
    
    return pwD, _mean, _std, _max, _min, _alpha, _beta, _alpha_beta_etc_values


'''
# P7 - calculate euclidean distance between aggCluster members
these are held in a dict of dataframes: cluster_pwDist[0] - [4]

This gives us a measure of the distances between aggCluster members per instance
The instances with the lowest average distances to other aggCluster members will
be centroids
Per aggCluster, rank the instances in terms of mean distance
Take 2 instances from each aggCluster as initial labelled set, L

'''

# must supply the D and chroma_X_df DataFrames as arguments
def cluster_euclidean_distances(_D, chroma_df, c):
    # first, make a dict that holds arrays of each aggCluster
    _cl_Boolean = {}
    _D_by_cluster = {} # ends up being a dict with 5 matrices shape (size of aggCluster, 4)
    _chroma_X_by_cluster = {}
    _cluster_pwDist = {}
    
    for i in np.unique(_D[c]): # c was '5clus'
        # returns a boolean true/false list we can use as an index
        # Looks through D, hauls out instances by aggCluster
        _cl_Boolean[i] = _D[c]==i
        # pick out aggCluster instances and have a separate structure for each
        # cl_Instances[x] lists StimID, Cat, Chroma and 5clus for each aggCluster
        _D_by_cluster[i] = _D[_cl_Boolean[i]]
        # pick out chroma_X data for each aggCluster
        _data = _chroma_X_by_cluster[i] = chroma_df[_cl_Boolean[i]]
        # make a dataframe for each aggCluster that returns pw_Dist for aggCluster members
        _cluster_pwDist[i] = pd.DataFrame(data=(euclidean_distances(_data, _data)),
                                     index=_chroma_X_by_cluster[i].index)
    
        print(_D_by_cluster[i].shape, _chroma_X_by_cluster[i].shape, _cluster_pwDist[i].shape)
        
    return _cl_Boolean, _D_by_cluster, _chroma_X_by_cluster, _cluster_pwDist


# must supply the D and chroma_X_df DataFrames as arguments
def cluster_euclidean_distances_II(chroma_df, c):
    # first, make a dict that holds arrays of each aggCluster
    _cl_Boolean = {}
    _D_by_cluster = {} # ends up being a dict with 5 matrices shape (size of aggCluster, 4)
    _chroma_X_by_cluster = {}
    _cluster_pwDist = {}
    
    for i in np.unique(c): # c was '5clus'
        # returns a boolean true/false list we can use as an index
        # Looks through D, hauls out instances by aggCluster
        _cl_Boolean[i] = c==i
        # pick out aggCluster instances and have a separate structure for each
        # cl_Instances[x] lists StimID, Cat, Chroma and 5clus for each aggCluster
        _D_by_cluster[i] = _cl_Boolean[i]
        # pick out chroma_X data for each aggCluster
        _data = _chroma_X_by_cluster[i] = chroma_df[_cl_Boolean[i]]
        # make a dataframe for each aggCluster that returns pw_Dist for aggCluster members
        _cluster_pwDist[i] = pd.DataFrame(data=(euclidean_distances(_data, _data)),
                                     index=_chroma_X_by_cluster[i].index)
    
        print(_D_by_cluster[i].shape, _chroma_X_by_cluster[i].shape, _cluster_pwDist[i].shape)
        
    return _cl_Boolean, _D_by_cluster, _chroma_X_by_cluster, _cluster_pwDist

'''
# Identify aggCluster centroids
# 
# check this logic:
# cluster_pwDist provides matrices which contain pairwise distance in a 
# (aggCluster size, aggCluster size) structure.
# centroids should be the instances that have the lowest mean distance to all 
# other aggCluster points
# so sort the means for each instance, take the lowest means as centroids.

# convolves a dict of dataframes that list the mean distances per agglomerative cluster
# need to supply cluster_pwDist as the argument to this function

# RUN THIS TO RESET FOR TESTING
'''

# make an order to get instances from to make sure extra instances come from biggest aggCluster
cluster_order = [0, 3, 1, 2, 4]

# boolean switch to control if we take an instance from this aggCluster
cluster_Empty = [False, False, False, False, False]

# COMMENT THIS BACK IN IF STARTING FROM SCRATCH
# cluster_means_dfs = {} # df of mean distances by aggCluster

# =============================================================================
# count = 0 # global variables declared here so they can be accessed in multiple functions
# cl_count = 0 # see: load_initial_instances() and get_instance_pwDist()
# =============================================================================


def aggCluster_mean_distances(_cluster_pwDist):
    
    # global cluster_means_df # access global variable
    
    _cluster_means_dict = {}

    # constitute aggClusters for mean distances in DataFrames
    # I can delete them from the DataFrames and preserve the index values for other slots
    for i in _cluster_pwDist:

        # calculate mean pw distance for each instance
        _cluster_means_dict[i] = pd.DataFrame(data=np.mean(_cluster_pwDist[i], axis=1),
                                           # use index from cluster_pwDist DataFrames
                                          index=_cluster_pwDist[i].index)

        # _cluster_means_dict[cluster number][only one column][no of instances in cluster]
        print(_cluster_means_dict[i].shape)
        
    return _cluster_means_dict


# get NoL new instances, put indices for them into S1_index
def load_aggCluster_initInstances(no_instances, _cluster_means_dfs):
    
    print("RUNNING")
    global count
    global cl_count
    global cluster_Empty
    global cluster_order
    
    _S1_index = []
    #_cluster_means_dfs_drop = {} # df of mean distances by aggCluster
    
    count = 0 # track instances got
    cl_count = 0 # track which aggCluster to look at
    
    print("count = ", count, "cl_count = ", cl_count)
    
    while count < no_instances:
        _cluster_centroid_idx, _cmDFs = get_aggCluster_instance_pwDist(cl_count, _cluster_means_dfs)
        
        #clus_Focus = cluster_order[cl_count]
        
        # add this index to an array that we'll use to conform training and test data later
        # and load it into S1
        _S1_index.append(_cluster_centroid_idx)
        
        # delete these instances from aggCluster_means_df
        #_cluster_means_dfs[clus_Focus] = _cluster_means_dfs[clus_Focus].drop([_cluster_centroid_idx], axis=0)
        
        
        # Flip boolean switch for this aggCluster if it's now empty
        #if _cluster_means_dfs[clus_Focus].shape[0] < 1:
        #    cluster_Empty[clus_Focus] = True
        
        count += 1
        cl_count += 1
        
        
        if cl_count > 4:
            cl_count = 0
        
        
        #if all(cluster_Empty):
        #    print("All aggClusters empty!")
        #    break
        
    return _S1_index, _cmDFs

            
# get a single instance
def get_aggCluster_instance_pwDist(cluster_no, _cluster_means_dfs):
    
    # provide access to global variables
    global count
    global cl_count
    #global S1_index
    
    global cluster_Empty
    global cluster_order
    
    clus_Focus = cluster_order[cluster_no]
    
    if cluster_Empty[clus_Focus]==False:
        
        # if this aggCluster is not empty
        # per aggCluster, get the index of the lowest mean distance value
        # these will be centroids for that aggCluster
        # print("#### aggCluster NUMBER ####", clus_Focus)
        
        # find the index of the smallest mean value in this column
        # (contains mean distance per instance to other instances)
        cluster_centroid_loc = _cluster_means_dfs[clus_Focus].values.argmin()
        
        # use this to get the index of the instance in all other data structures
        _cluster_centroid_idx = _cluster_means_dfs[clus_Focus].index[cluster_centroid_loc]
        
        ##print("Loc = ", cluster_centroid_loc)
        ##print("Idx = ", _cluster_centroid_idx)
        
        # add this index to an array that we'll use to conform training and test data later
        # and load it into S1
        #S1_index.append(_cluster_centroid_idx)

        # print("Found instance, S1_index now length: ", len(S1_index))

        # delete these instances from aggCluster_means_df
        _cluster_means_dfs[clus_Focus] = _cluster_means_dfs[clus_Focus].drop([_cluster_centroid_idx], axis=0)
        # print("_cluster_means_dfs shape for cluster: ", clus_Focus, _cluster_means_dfs[clus_Focus].shape)

        # Flip boolean switch for this aggCluster if it's now empty
        #if _cluster_means_dfs[clus_Focus].shape[0] < 1:
        #    cluster_Empty[clus_Focus] = True
        
        #count += 1
        #cl_count += 1
        
        #print(S1_index)
        
        return _cluster_centroid_idx, _cluster_means_dfs, 
        
    #else:
        #cl_count += 1
    
