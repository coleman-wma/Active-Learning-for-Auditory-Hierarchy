#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:11:43 2019

@author: billcoleman
"""

'''
To load data from pickle file into environment, if the file reads aren't nested
they crap out.
'''

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler # to normalise data

def testing_func(x):
    x = x * 5
    return x

def load_data():
    # Load all dataset
    with open('data/all_stimInfo_FINAL_20190508.data', 'rb') as filehandle:  
        # read the data as binary data stream
        _D_all = pickle.load(filehandle)
    
        # Load aggCluster information
        with open('data/clusterData_20190511.data', 'rb') as filehandle2:  
            # store the data as binary data stream
            _Cl_all = pickle.load(filehandle2)
        
    return _D_all, _Cl_all


def load_data_jul():
    # Load all dataset
    
    with open('data/allDATA_JULY_20190711.data', 'rb') as filehandle:  
        # read the data as binary data stream
        newdata = pickle.load(filehandle)
        newdata = pd.DataFrame(newdata)
    
    return newdata

D_all_jul = load_data_jul()


'''
Make a dataframe, D, that holds just the information we're interested in
'''

def make_D(df1, df2):
    # concat one DF that holds just the data we're going to work with
    _D = pd.concat([df1['StimID'], df1['Cat'], df1['chroma_Del'], df2['5clus'], \
                    df2['8clus'], df2['10clus']], 
                 axis=1,
                 join='outer')
    
    return _D


'''
I'm interested in the max and min of these values because notionally it's best 
to be working with normalised values. The chroma data has a max of 1 and a min 
very close to 0 so I've not normalised it.
'''

def find_max_min(D):
    
    this_max = 0
    this_min = 1000
    
    for i in range(0, D.shape[0]):
        
        if D['log_p_mel_spec'][i].shape[1] > this_max:  # was 'chroma_Del' .max()
            this_max = D['log_p_mel_spec'][i].shape[1]
            
        if D['log_p_mel_spec'][i].shape[1] < this_min:  # .min()
            this_min = D['log_p_mel_spec'][i].shape[1]
            
        return this_max, this_min
    
'''
# See EGAL_I_Dendrograms_Clustering for the nuts and bolts of this \/ \/ \/ 

# current shape of the chroma data in D is a Dataframe for each instance
# of 12 rows and >156 columns

# the following code
# clamp max length to 156
# look at each entry 
# take values 0 - 156 from array
# move to next array
# repeat

# this basically flattens the 12 x ~156 arrays into one long one
'''

def flatten_chroma(D):
    
    ch_X = []

    for i in range(0, D.shape[0]):
        a = D['chroma_Del'][i]
        a = np.resize(a,(12, 156)) # deal with instances that are longer than 156
        a_long = np.reshape(a, 1872) # flatten the result into one vector
        ch_X.append(a_long) # add it to array
    
    scaler = MinMaxScaler() # scale data
    ch_X_scaled = scaler.fit_transform(ch_X)
    
    return ch_X_scaled


def flatten_log_p_mel_spec(D):
    
    ch_X = []

    for i in range(0, D.shape[0]):
        a = D['log_p_mel_spec'][i]
        # Each entry is 128 bands x ? windows, ranges from 313 to 453
        # Clamp to 313
        a = np.resize(a, (128, 313))
        a_long = np.reshape(a, 40064) # flatten the result into one vector
        ch_X.append(a_long) # add it to array
    
    # I'm not normalising this because they don't appear to have done so in the
    # Salamon & Bello paper - something to do with log_power features maybe?
    
    return ch_X


def flatten_log_p_mel_spec_40bands(D):
    
    ch_X = []

    for i in range(0, D.shape[0]):
        a = D['lpms_data'][i]
        # Each entry is 40 bands x 156 windows, ranges from 313 to 453
        # Clamp to 313
        a = np.resize(a, (40, 156))
        a_long = np.reshape(a, 6240) # flatten the result into one vector
        ch_X.append(a_long) # add it to array
    
    # I'm not normalising this because they don't appear to have done so in the
    # Salamon & Bello paper - something to do with log_power features maybe?
    
    return ch_X


feat_list = ['mfcc', 'mfcc_Del', 'mfcc_Del2', 'mfcc_Del5', 'chroma',
       'chroma_Del', 'chroma_Del2', 'chroma_Del5']

with open('data/all_LPMS_40bands_20190709.data', 'rb') as filehandle:  
        # read the data as binary data stream
        log_pow_mel_spec_40 = pickle.load(filehandle)

# log_pow = flatten_log_p_mel_spec(log_pow_mel_spec)

# log_pow_40 = flatten_log_p_mel_spec_40bands(log_pow_mel_spec_40)

def flatten_data(D): # D here should be D_all
    
    D_data_dict = {}
    count_int = 0
    
    for j in feat_list:
    
        ch_X = []
    
        for i in range(D.shape[0]):
            a = D[j][i]
            # deal with instances that are longer than 156
            a = np.resize(a, (D[j][i].shape[0], 156))
            # account for different number of bins in mfcc (12) and chroma (13)
            a_len = D[j][i].shape[0] * 156
            # flatten the result into one vector
            a_long = np.reshape(a, a_len)
            # add it to array
            ch_X.append(a_long) 
        
        scaler = MinMaxScaler() # scale data
        ch_X_scaled = scaler.fit_transform(ch_X)
        
        print(ch_X_scaled.shape)
        print(count_int)
        
        D_data_dict[count_int] = ch_X_scaled
        
        count_int += 1

    return D_data_dict


# Flatten initial data structures. D_all is generated from load_data()
#D_data_dict = flatten_data(D_all)


def super_dataVector(dataDict):
    
    final_vector = []
    
    # loop through all data data vectors
    # and concatenate the different features together

    for j in range(3002):
    
        thisRow = dataDict[0][j].tolist()
        
        for i in range(1, 8):
            thisRow.extend(dataDict[i][j].tolist())
        
        final_vector.append(thisRow)
        
    return final_vector


# munging all the features together in one    
#feats_Concat = super_dataVector(D_data_dict)

# Add it to the other data in a dict
#D_data_dict[8] = np.array(feats_Concat)


def best_featsets(all_featureSets):
    
    mfccchroma_0 = []
    
    for j in range(3002):
    
        newRow = all_featureSets[0].iloc[j].tolist()
        newRow.extend(all_featureSets[4].iloc[j].tolist())
        mfccchroma_0.append(newRow)
    
    return mfccchroma_0


# mfcc_plus_chroma = best_featsets(all_featureSets)


def new_feature_dict(all_featureSets, mfcc_plus_chroma):
    
    new_feat_vecs = {}
    
    new_feat_vecs[0] = all_featureSets[0]
    new_feat_vecs[1] = all_featureSets[4]
    new_feat_vecs[2] = pd.DataFrame(mfcc_plus_chroma)
    
    return new_feat_vecs


# focus_features_dict = new_feature_dict(all_featureSets, mfcc_plus_chroma)


def new_feature_dict_II(all_featureSets, log_pow, log_pow_40):
    
    new_feat_vecs = {}
    
    new_feat_vecs[0] = pd.DataFrame(all_featureSets[0])
    new_feat_vecs[1] = pd.DataFrame(all_featureSets[8])
    new_feat_vecs[2] = pd.DataFrame(log_pow)
    new_feat_vecs[3] = pd.DataFrame(log_pow_40)
    
    return new_feat_vecs


# =============================================================================
# focus_features_dict_new = new_feature_dict_II(all_featureSets,
#                                               log_pow,
#                                               log_pow_40)
# =============================================================================


def flat_mfcc(data_vec):
    
    flat_data = []

    for i in range(0, data_vec.shape[0]):
        a = data_vec[i]
        a = np.resize(a, (13, 156)) # deal with instances that are longer than 156
        a_long = np.reshape(a, 2028) # flatten the result into one vector
        flat_data.append(a_long) # add it to array
    
    scaler = MinMaxScaler() # scale data
    flat_data_scaled = scaler.fit_transform(flat_data)
    
    return flat_data_scaled


def flat_chroma(data_vec):
    
    flat_data = []

    for i in range(0, data_vec.shape[0]):
        a = data_vec[i]
        a = np.resize(a, (12, 156)) # deal with instances that are longer than 156
        a_long = np.reshape(a, 1872) # flatten the result into one vector
        flat_data.append(a_long) # add it to array
    
    scaler = MinMaxScaler() # scale data
    flat_data_scaled = scaler.fit_transform(flat_data)
    
    return flat_data_scaled


def flat_lpms(data_vec):
    
    flat_data = []

    for i in range(0, data_vec.shape[0]):
        a = data_vec[i]
        a = np.resize(a, (40, 156)) # deal with instances that are longer than 156
        a_long = np.asarray(np.reshape(a, 6240)) # flatten the result into one vector
        flat_data.append(a_long) # add it to array
        array_2d = np.array(flat_data)
    
    return array_2d


def flat_lpms_scaled(data_vec):
    
    flat_data = []

    for i in range(0, data_vec.shape[0]):
        a = data_vec[i]
        a = np.resize(a, (40, 156)) # deal with instances that are longer than 156
        a_long = np.reshape(a, 6240) # flatten the result into one vector
        flat_data.append(a_long) # add it to array
    
    scaler = MinMaxScaler() # scale data
    flat_data_scaled = scaler.fit_transform(flat_data)
    
    return flat_data_scaled


# Supply D_all_jul to this function
def flat_feats_jul(df):
    
    _flat_df = {}
    
    for i in df.columns:
        
        no_i = df.columns.get_loc(i)
        
        if (no_i > 1 and no_i < 6):
            _flat_df[i] = flat_mfcc(df[i])
            print("mfccs done!", i)
        
        if (no_i > 5 and no_i < 10):
            _flat_df[i] = flat_chroma(df[i])
            print("chromas done!", i)
        
        if (no_i > 9):
            _flat_df[i] = flat_lpms(df[i])
            print("lpms done!", i)

        if (no_i > 9):
            _flat_df[i + '_scaled'] = flat_lpms_scaled(df[i])
            print("lpms scaled done!", i)
    
    return _flat_df

#all_featureSets_jul = flat_feats_jul(D_all_jul)


def convert_to_df(df):
    
    for key in df:
        df[key] = pd.DataFrame(df[key])
    
    return df

#all_featureSets_jul = convert_to_df(all_featureSets_jul)

def save_allFeats():
    
    # global D_data_dict
    global all_featureSets_jul

    print("================= Saving FEATURE Files ==================")
        
# =============================================================================
#     with open('data/FEATURES_concat_feats.data', 'wb') as allFeats:  
#         # store the data as binary data stream
#         pickle.dump(D_data_dict, allFeats)
# =============================================================================
        
    with open('data/FEATURES_JULY_mfcc_chroma_lpms.data', 'wb') as allFeats:  
        # store the data as binary data stream
        pickle.dump(all_featureSets_jul, allFeats)


# Finding the length of feature vectors
def find_len_featVecs():
    
    for i in range(3002):

           x0 = log_pow_mel_spec_40['lpms_data'][i].shape[0]
           shape0.append(x0)
           
           x1 = log_pow_mel_spec_40['lpms_data'][i].shape[1]
           shape1.append(x1)
