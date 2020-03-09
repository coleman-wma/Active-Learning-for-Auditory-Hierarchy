#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:15:03 2019

@author: billcoleman
"""

'''

###############################################
########### EGAL IMPLEMENTATION ###############
###############################################

CALCULATE DENSITY MEASURE FOR EVERYTHING
The sum of the distances from each instance to all other instances within radius alpha

CALCULATE DIVERSITY MEASURE FOR EVERY INSTANCE NOT IN S1
The distance between each unlabelled instance and its closest labelled neighbour (member of S1)

CALCULATE CANDIDATE SET (CS)
All unlabelled instances where diversity measure is greater than beta

'''

import trainTest_functions

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances#, cosine_similarity
from sklearn.svm import SVC # SVM model


#################
#### DENSITY #### "The sum of the distances from the instance to all other
#################        instances within radius alpha."

# returns the sum of all the similarities where the similarity value <= alpha
def get_sum_of_density_values(similarities_i, _alpha_):

    #global alpha_

    """
    similarities_i: similarity matrix - my equivalent of this is pairwiseDist
    alpha: threshold of the density radius
    """
    # check the instances conform to the rule first
    # return similarities_i[similarities_i <= alpha]
    # then return the density for the instance
    return np.sum(similarities_i[similarities_i <= _alpha_])


def get_sorted_density_values(_pairwiseDist, _U, _alpha_):
    
    # construct a dict to hold density values for each instance its required for
    _density_dict = {}

    for i in range(len(_U)):

        # feed each row in pairwiseDist into a function that filters for values
        # <= alpha
        _density_dict[_U[i]] =\
            get_sum_of_density_values(_pairwiseDist[i, :], _alpha_)

    # sort the final dict by values so that the smallest values are first and 
    # the largest last
    # NOTE: this returns a number of rows with zeros - outside the range of
    # alpha presumably
    # Using euclidean distance means that the smallest values > 0 are the most
    # dense instances so excluding those instances with zero, I want the 
    # smallest values

    # NOTE: there are 108 items with a value of 0 in this dict
    # may not actually need this because these may get filtered out in
    # find_candidate_set()
    _denseVals_sort = dict(sorted(_density_dict.items(),
                                  key=lambda kv: kv[1],
                                  reverse=False))

    # makes a dict of the above removing the zeros
    # may not actually need this because these may get filtered out in
    # find_candidate_set()
    # _denseVals_sort_noZeros =\
    # dict((k,v) for k,v in _denseVals_sort.items() if v != 0)

    return _denseVals_sort, _density_dict#, _denseVals_sort_noZeros   


###################
#### DIVERSITY #### "The distance between the unlabelled instance and its
###################    nearest labelled neighbour."

# Doing another version of this using dataframes for greater transparency
def find_candidate_set_vII(L_,
                           U_,
                           density_dict,
                           U,
                           beta_,
                           beta_old,
                           NoL,
                           w,
                           stop_all,
                           loop_counter):
    
    # find distance values between labelled and unlabelled sets
    # np.asanyarray converts the list into an array ## WESAM
    # s = euclidean_distances(np.asanyarray(self.train_X, dtype=np.float32),
    # self.pool_X) ## WESAM
    s = euclidean_distances(L_, U_)
    
    # not dividing 1 by the result here because I can take care of that in the
    # next step taking the minimum here because I need the distance to the
    # NEAREST labelled neighbour
    # div is my equivalent of Wesam's diversity_inversed
    div = np.min(s, axis=0)
    
    # make a dict from density_dict for all instances in U and div
    
    '''
    # where div > beta - THIS IS A CHANGE FROM WESAMS CODE
    # we should be filtering for diversity values outside the radius because
    # we're interested in focusing on getting labels from more diverse 
    # regions those that are FURTHER AWAY from nearest labelled example
    '''
    
    # making a dataframe that holds density and diversity values per instance
    # in U
    
    filtered_dict = list(filter(lambda item: item[0] in U,
                                density_dict.items())) # <<<<<<
    filtered_dict_dense = [i[1] for i in filtered_dict] # strips away index
    filtered_dict_idx = [i[0] for i in filtered_dict]
    
    make_dict = {'dense':filtered_dict_dense,
                 'check_idx':filtered_dict_idx,
                 'U':U,
                 'diverse':div}
    _df_denseDiv = pd.DataFrame(make_dict, index=U)
    
    
    if w == 0:
        
        print("w = 0 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ PURE DIVERSITY")
        
        # if w = 0 then selection is purely diversity based, so take the 
        # largest diversity values
        cs_df = _df_denseDiv
        cs_df_idx = cs_df.nlargest(NoL, 'diverse', keep='first')
    
    elif w > 0 and w < 1: # != 1:
        
        print("w = ", w, " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        
        # Filtering to Candidate Set where diverse value falls within bounds 
        # controlled by beta_
        cs_df = _df_denseDiv[(_df_denseDiv.diverse > beta_) &\
                             (_df_denseDiv.diverse <= beta_old)]
        
        # Find the NoL instances with the largest density values
        # the index of this dataframe is the index for instances to take from
        # U and put into L
        if cs_df.shape[0] > NoL:
            
            '''
            # this may be another misunderstanding from the paper caused by
            # using euclidean versus cosine similarity/distance
            # The Hu paper talks about taking instances with 'highest' density
            # for labelling first. In the context of cosine similarity, these 
            # instances are those most similar to each other - so the 
            # equivalent in a euclidean distance context would be instances 
            # with the lowest density values.
            # 
            # It intuitively makes sense if we're using the diversity value to
            # factor in a sense of difference, that density should factor in a
            # sense of sameness. Instances with higher density values are 
            # closer together - they are not outliers.
            '''
            
            cs_df_idx = cs_df.nlargest(NoL, 'dense', keep='first')
            # cs_df_idx = cs_df.nsmallest(NoL, 'dense', keep='first')
        
        else:
            cs_df_idx = cs_df
            
            # update beta_
            beta_old = beta_
            beta_ = update_beta(beta_, NoL, w, div)
            
            print("beta_ changed to: ", beta_, "<<<<<<<<<<<<<<<<<<<<<<<<")
            
            if beta_old == beta_:
                print("Beta_ isn't changing. %%%%%%%%%%%%%%%%%%%%%%%%%")
                stop_all = True
    
    else:
        print("w = ", w, " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ PURE DENSITY")
        # when w = 1 we don't need to control for beta_
        # so just grab the next NoL instances with the smallest density values
        # in _df_denseDiv
        cs_df = _df_denseDiv
        cs_df_idx = cs_df.nlargest(NoL, 'dense', keep='first')
        # cs_df_idx = cs_df.nsmallest(NoL, 'dense', keep='first')
    
    return div, s, _df_denseDiv, cs_df, cs_df_idx, beta_, stop_all,\
        loop_counter #, _df_dense_idx_diverse


# Doing another version of this using dataframes for greater transparency
def find_candidate_set_vIII(L_,
                           U_,
                           density_dict,
                           U,
                           beta_,
                           beta_old,
                           NoL,
                           w,
                           stop_all,
                           loop_counter,
                           S1_index, 
                           chroma_X_df):
    
    # find distance values between labelled and unlabelled sets
    # np.asanyarray converts the list into an array ## WESAM
    # s = euclidean_distances(np.asanyarray(self.train_X, dtype=np.float32),
    # self.pool_X) ## WESAM
    s = euclidean_distances(L_, U_)
    
    # not dividing 1 by the result here because I can take care of that in the
    # next step taking the minimum here because I need the distance to the
    # NEAREST labelled neighbour
    # div is my equivalent of Wesam's diversity_inversed
    div = np.min(s, axis=0)
    
    # make a dict from density_dict for all instances in U and div
    
    '''
    # where div > beta - THIS IS A CHANGE FROM WESAMS CODE
    # we should be filtering for diversity values outside the radius because
    # we're interested in focusing on getting labels from more diverse 
    # regions those that are FURTHER AWAY from nearest labelled example
    '''
    
    # making a dataframe that holds density and diversity values per instance
    # in U
    
    filtered_dict = list(filter(lambda item: item[0] in U,
                                density_dict.items())) # <<<<<<
    filtered_dict_dense = [i[1] for i in filtered_dict] # strips away index
    filtered_dict_idx = [i[0] for i in filtered_dict]
    
    make_dict = {'dense':filtered_dict_dense,
                 'check_idx':filtered_dict_idx,
                 'U':U,
                 'diverse':div}
    _df_denseDiv = pd.DataFrame(make_dict, index=U)
    
    
    if w == 0:
        
        print("w = 0 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ PURE DIVERSITY")
        
        # if w = 0 then selection is purely diversity based, so take the 
        # largest diversity values
        cs_df = _df_denseDiv
        cs_df_idx = cs_df.nlargest(NoL, 'diverse', keep='first')
    
    elif w > 0 and w < 1: # != 1:
        
        print("w = ", w, " $$$$$$$$$$$$$ BTW 0 and 1 $$$$$$$$$$$$4$$$$$")
        
        # Filtering to Candidate Set where diverse value falls within bounds 
        # controlled by beta_
        cs_df = _df_denseDiv[(_df_denseDiv.diverse > beta_) &\
                             (_df_denseDiv.diverse <= beta_old)]
        
        # Find the NoL instances with the largest density values
        # the index of this dataframe is the index for instances to take from
        # U and put into L
        if cs_df.shape[0] >= NoL:
            
            '''
            # this may be another misunderstanding from the paper caused by
            # using euclidean versus cosine similarity/distance
            # The Hu paper talks about taking instances with 'highest' density
            # for labelling first. In the context of cosine similarity, these 
            # instances are those most similar to each other - so the 
            # equivalent in a euclidean distance context would be instances 
            # with the highest density values.
            # 
            # It intuitively makes sense if we're using the diversity value to
            # factor in a sense of difference, that density should factor in a
            # sense of sameness. Instances with higher density values are 
            # closer together - they are not outliers.
            '''
            
            cs_df_idx = cs_df.nlargest(NoL, 'dense', keep='first')
            # cs_df_idx = cs_df.nsmallest(NoL, 'dense', keep='first')
        
        else:
            mini_NoL = NoL - cs_df.shape[0]
            incomplete_batch = cs_df
            
            if mini_NoL > len(U):
                mini_NoL = len(U)
                print("THIS IS THE LAST BATCH OF INSTANCES TO BE LABELED")
            
            # update beta_
            beta_old = beta_
            beta_ = update_beta(beta_, NoL, w, div)
            
            print("beta_ changed to: ", beta_, "<<<<<<<<<<<<<<<<<<<<<<<<")
            print("Mini batch size = ", mini_NoL, "< < < < < < < < < < < < < < < < <<<<<<<<")
            
            # get minibatch to add to cs_df to keep batch sizes uniform
            # beta is updated so:
            # remove instances in incomplete_batch from what you supply to next
            # function
            # setup temporary L_ and U_ so we can fill this batch
            S1_temp = S1_index.copy()
            S1_temp.extend(incomplete_batch.index)
            mini_L_data = chroma_X_df.loc[S1_temp]  # switched to loc here
            
            mini_U_idx = U.copy()
            # remove these instances from U
            for j in incomplete_batch.index:
                # delete the element by value NOT index
                mini_U_idx.remove(j)
            
            mini_U_data = chroma_X_df.loc[mini_U_idx]  # switched to loc here
            
            print("Shape of mini_L_data = ", mini_L_data.shape,
                  "Shape of mini_U_data = ", mini_U_data.shape,
                  "Length of mini_U_idx = ", len(mini_U_idx))
            
            # Send mini_L_ and mini_U_ to function to find candidate set for
            # this minibatch if there are instances in the minibatch
            
            if len(mini_U_idx) < 1:
                
                cs_df_idx = incomplete_batch
                
            else:
                
                mini_batch_instances = complete_this_batch(mini_L_data,
                                                           mini_U_data,
                                                           density_dict,
                                                           mini_U_idx,
                                                           beta_,
                                                           beta_old,
                                                           mini_NoL)
                
                cs_df_idx = pd.DataFrame(pd.concat([incomplete_batch, mini_batch_instances],
                                      # make sure no duplicate index values
                                      verify_integrity=True))
            
            if beta_old == beta_:
                print("Beta_ isn't changing. %%%%%%%%%%%%%%%%%%%%%%%%%")
                stop_all = True
    
    else:
        print("w = ", w, " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ PURE DENSITY")
        # when w = 1 we don't need to control for beta_
        # so just grab the next NoL instances with the smallest density values
        # in _df_denseDiv
        cs_df = _df_denseDiv
        cs_df_idx = cs_df.nlargest(NoL, 'dense', keep='first')
        # cs_df_idx = cs_df.nsmallest(NoL, 'dense', keep='first')
    
    return div, s, _df_denseDiv, cs_df, cs_df_idx, beta_, stop_all,\
        loop_counter


# Doing another version of this using dataframes for greater transparency
def complete_this_batch(L_,
                           U_,
                           density_dict,
                           U,
                           beta_,
                           beta_old,
                           NoL):
    
    # find distance values between labelled and unlabelled sets
    # np.asanyarray converts the list into an array ## WESAM
    # s = euclidean_distances(np.asanyarray(self.train_X, dtype=np.float32),
    # self.pool_X) ## WESAM
    s = euclidean_distances(L_, U_)
    
    # not dividing 1 by the result here because I can take care of that in the
    # next step taking the minimum here because I need the distance to the
    # NEAREST labelled neighbour
    # div is my equivalent of Wesam's diversity_inversed
    div = np.min(s, axis=0)
    
    # make a dict from density_dict for all instances in U and div
    
    '''
    # where div > beta - THIS IS A CHANGE FROM WESAMS CODE
    # we should be filtering for diversity values outside the radius because
    # we're interested in focusing on getting labels from more diverse 
    # regions those that are FURTHER AWAY from nearest labelled example
    '''
    
    # making a dataframe that holds density and diversity values per instance
    # in U
    
    filtered_dict = list(filter(lambda item: item[0] in U,
                                density_dict.items())) # <<<<<<
    filtered_dict_dense = [i[1] for i in filtered_dict] # strips away index
    filtered_dict_idx = [i[0] for i in filtered_dict]
    
    make_dict = {'dense':filtered_dict_dense,
                 'check_idx':filtered_dict_idx,
                 'U':U,
                 'diverse':div}
    _df_denseDiv = pd.DataFrame(make_dict, index=U)
    
    cs_df = _df_denseDiv[(_df_denseDiv.diverse > beta_) &\
                             (_df_denseDiv.diverse <= beta_old)]
    
    mini_batch_instances = cs_df.nlargest(NoL, 'dense', keep='first')
    # mini_batch_instances = cs_df.nsmallest(NoL, 'dense', keep='first')
    
    return mini_batch_instances

'''
After running this function CS_sort_filt will hold an index to the next 
instances to be used for labelling. Take the NoL first instances as these are
the densest, most diverse (outside radius of beta_) there are.

Update S1. Retrain model. Log score for plot. Time to get more labels.

Recalculate CS - because we have new instances which will mean the diversity
measure for possible candidates will change. Take the densest instances of the
new CS for labelling.

Keep repeating this until there are no instances in CS.

Adjust beta_. Recalculate CS.

Repeat until beta_stops changing. Stop process.
'''


# =============================================================================
# def printValues(offset):
#     for f in range(50):
#         print(CS[CS_sort[f + offset]])
# =============================================================================

        
def update_beta(beta_, NoL, w, diversity_values_U_L):#, loop_counter):

    #global track_beta

    ''' 
    When w = 0, EGAL defaults to pure diversity based
    When w = 1, EGAL defaults to pure density based
    '''

    # the numeric in the [::1] controls the order of the sort
    # slice notation [start here, end here, step (or order)]

    # change numeric to -1 for reverse order
    # adding .sort() on the end sorts
    diversity_values_U_L[::-1].sort()

    # sw roughly gives the index for the slot in the structure that will have 
    # new beta_

    #if w == 1: # when w is 1 it will index outside the bounds of U unless we correct
        # cnvert diversity_values_U_L to a list, find the index of the max value
        #_sw = diversity_values_U_L.tolist().index(np.max(diversity_values_U_L))
        #_sw = len(diversity_values_U_L) - (NoL + 1)
        
    #else:
    _sw = w * (len(diversity_values_U_L) + 1)
    
    if _sw < NoL:
        # let beta_ equal to 0
        # because there are very few instances left in U
        beta_ = 0
        #loop_counter = 500
        #track_beta.append(beta_)
    
    else:
        # let beta_ equal to the value in the slot that splits instances
        # in U to the proportion dictated by w
        beta_ = diversity_values_U_L[round(_sw)]
        #track_beta.append(beta_)
        
        if beta_ is None:
            beta_ = 0
            #loop_counter = 500
            print("########## beta_ POPPED TO None, RESET to 0 #############")
    
    return beta_#, loop_counter #, _sw

        

# Use EGAL algorithm to select instances for labelling
def egal_loop_vII(loop_counter,
                  stop_all, 
                  U, 
                  S1_index, 
                  chroma_X_df,
                  D,
                  L,
                  y_true_list,
                  y_pred_list,
                  trainAcc,
                  testAcc,
                  labelled_instances,
                  density_dict,
                  beta_,
                  beta_old,
                  NoL,
                  w,
                  ft,
                  al,
                  y_pred_dec_list,
                  test_data):
    
    # convert U to a list so I can use .remove()
    U = U.tolist()

    # while there are instances in diversity_values_U_L and stop_all is True
    while len(U) > 0 and not stop_all: 
        # previously used len(S1_index) < D.shape[0] + 1
        # previously used diversity_values_U_L) > 0
        # in place of diversity_values_U_L

        print("EGAL LOOP: ", loop_counter, " ::: Feature Set: ", ft, " ::: Alpha Method: ", al)
        print("--------------------------------------------------------------")

        # update diversity values
        diversity_values_U_L, eucl_distanceS_U_L, CS, CS_sort,\
        CS_sort_filt, beta_, stop_all, loop_counter =\
        find_candidate_set_vIII(chroma_X_df.loc[S1_index],  # vIII maintains cohesive batch sizes
                               chroma_X_df.loc[U],  # switched from iloc to loc for both of these
                               density_dict, 
                               U,
                               beta_,
                               beta_old,
                               NoL,
                               w, 
                               stop_all,
                               loop_counter,
                               S1_index, 
                               chroma_X_df)
        
        
        
        # update S1
        S1_index.extend(CS_sort_filt.index)
        print("Number of labels being added = ", CS_sort_filt.shape[0])
        print("Size of S1_index now = ", len(S1_index))
        
        # remove these instances from U - wasn't working in train_Test()
        for j in CS_sort_filt.index:
            # delete the element by value NOT index
            U.remove(j)
        
        print("Number of unlabelled instances, U = ", len(U))
        
        if len(U) < 1:
            stop_all = True
        
        if stop_all:
            print("STOP ALL - Getting one last score - loop = ", loop_counter)
            
# =============================================================================
#         if testAcc[-1] < 0.9 and len(S1_index) < 1500:
# =============================================================================
        
        #if loop_counter % 20 == 0:
            
        print("--------------------------------------------------------------")
        print(":::::::::::::::::::::: Getting a score :::::::::::::::::::::::")
        print("--------------------------------------------------------------")
        
        # train test        
        U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
            labelled_instances, y_pred_dec_list = \
            trainTest_functions.train_Test_EGAL(chroma_X_df,
                                           S1_index, 
                                           D, 
                                           U, 
                                           L, 
                                           loop_counter, 
                                           y_true_list,
                                           y_pred_list,
                                           trainAcc,
                                           testAcc,
                                           labelled_instances,
                                           y_pred_dec_list,
                                           test_data)
                
# =============================================================================
#                 if testAcc[-1] < 0.7 and len(S1_index) > 800:
#                     # then this run isn't preforming well at all
#                     # set NoL=500 to expedite run
#                     print("This run is performing poorly, so lets EXPEDITE. NoL=500")
#                     NoL=500
# =============================================================================
        
# =============================================================================
#         else:
#             print("Score now above 0.9 or len(S1_index) > 1500, EXPEDITE. NoL=500")
#             NoL = 500
#             
#             print("--------------------------------------------------------------")
#             print(":::::::::::::::::::::: Getting a score :::::::::::::::::::::::")
#             print("--------------------------------------------------------------")
#             
#             # train test        
#             U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
#             labelled_instances, y_pred_dec_list = \
#             trainTest_functions.train_Test_EGAL(chroma_X_df,
#                                                S1_index, 
#                                                D, 
#                                                U, 
#                                                L, 
#                                                loop_counter, 
#                                                y_true_list,
#                                                y_pred_list,
#                                                trainAcc,
#                                                testAcc,
#                                                labelled_instances,
#                                                y_pred_dec_list)
# =============================================================================
            
            
        
        if stop_all:
            break
        
        loop_counter = loop_counter + 1
        
    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, diversity_values_U_L, eucl_distanceS_U_L, CS,\
        CS_sort, CS_sort_filt, beta_, y_pred_dec_list
    
    print("########## EGAL Loop ENDED #############")


# Use EGAL algorithm to sleect instances for labelling
def EGAL_SL_loop(loop_counter,
                  stop_all, 
                  U, 
                  S1_index, 
                  chroma_X_df,
                  D,
                  L,
                  y_true_list,
                  y_pred_list,
                  trainAcc,
                  testAcc,
                  labelled_instances,
                  density_dict,
                  beta_,
                  beta_old,
                  NoL,
                  w,
                  ft,
                  al,
                  y_pred_dec_list,
                  sl_threshold):
    
    # convert U to a list so I can use .remove()
    U = U.tolist()

    # while there are instances in diversity_values_U_L and stop_all is True
    while len(U) > 0 and not stop_all: 
        # previously used len(S1_index) < D.shape[0] + 1
        # previously used diversity_values_U_L) > 0
        # in place of diversity_values_U_L

        print("EGAL_SL LOOP: ", loop_counter, " ::: Feature Set: ", ft, " ::: Alpha Method: ", al)
        print("--------------------------------------------------------------")

        # update diversity values
        diversity_values_U_L, eucl_distanceS_U_L, CS, CS_sort,\
        CS_sort_filt, beta_, stop_all, loop_counter =\
        find_candidate_set_vII(chroma_X_df.iloc[S1_index], 
                               chroma_X_df.iloc[U], 
                               density_dict, 
                               U,
                               beta_,
                               beta_old,
                               NoL,
                               w, 
                               stop_all,
                               loop_counter)

        print("Size CS_sort_filt = ", CS_sort_filt.shape)
        print("Size diversity_values_U_L = ", diversity_values_U_L.shape)
        print("Number of unlabelled instances, U = ", len(U)) #- in train_Test()
        
        # update S1
        S1_index.extend(CS_sort_filt.index)
        
        # remove these instances from U - wasn't working in train_Test()
        for j in CS_sort_filt.index:
            # delete the element by value NOT index
            U.remove(j)
            
            if len(U) == 0:
                stop_all = True
        
        if stop_all:
            loop_counter = 500
            print("Loop Counter Flipped to ", loop_counter)
        
        # train test        
        U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, y_pred_dec_list = \
        trainTest_functions.train_Test_EGAL_SL(chroma_X_df,
                                           S1_index, 
                                           D, 
                                           U, 
                                           L, 
                                           loop_counter, 
                                           y_true_list,
                                           y_pred_list,
                                           trainAcc,
                                           testAcc,
                                           labelled_instances,
                                           y_pred_dec_list)
        
# =============================================================================
#         if testAcc[-1] > 0.9:
#             print("testAcc > 90%, CHANGING NoL to 200. @@@@@@@@@@@@@@@@@@@@@@")
#             NoL = 200
# 
#         if testAcc[-1] > 0.95:
#             print("testAcc > 95%, CHANGING NoL to 500. @@@@@@@@@@@@@@@@@@@@@@")
#             NoL = 500
# =============================================================================
        
        if stop_all:
            break
        
# =============================================================================
#         if loop_counter == 5:
#             print("CHECKING CS CONTENT SPECIAL STOP")
#             break
# =============================================================================
        
        # These are the model params used elsewhere - confusion over
        # which runs have which model as the one just below was also used
        clf_egal_sl = SVC(kernel='rbf',
                      C=50,
                      gamma=1,
                      probability=True)
            
        # get the following:
        # train data from chroma_X_df index S1
        X_train_ = chroma_X_df.iloc[S1_index]
        # test data from chroma_X_df index everything not in S1
        X_test_ = chroma_X_df.drop(S1_index, axis=0)
        # train labels from D index S1
        y_train_ = D['self_pred'].iloc[S1_index]
        # test labels from D index everything not in S1            
        # y_test_ = D['self_pred'].drop(S1_index, axis=0) 
        
        y_train_ = y_train_.astype('int')
        # y_test_ = y_test_.astype('int')
        
        # Fit model
        clf_egal_sl.fit(X_train_, y_train_)
        
        # get class prediction probability scores in a df
        # use the index from X_test as the index here
        usc_proba_df_ = pd.DataFrame(clf_egal_sl.predict_proba(X_test_),
                                    index=X_test_.index)
        
        # limit the number of instances that can be tagged as FG to 20%
        get_fg = round(NoL * 0.2)
        get_nfg = NoL - get_fg
        
        
        # 80% instances that are most likely nonFG get labelled nonFG         
        idx_nfg = usc_proba_df_.nlargest(get_nfg, 0, keep='first')
        # Unless the prediction probability is < sl_threshold
        idx_nfg = idx_nfg[idx_nfg[0] > sl_threshold]

        # Track the largest and smallest probability values
        nfg_lrg = idx_nfg.nlargest(1, 0, keep='first')
        nfg_sml = idx_nfg.nsmallest(1, 0, keep='first')
        print("LARGEST nFG PROB: ", nfg_lrg[0])
        print("SMALLEST nFG PROB: ", nfg_sml[0])
        
        # Drop instances used in idx_fg
        idx_fg = usc_proba_df_.drop(idx_nfg.index, axis=0)
        # 20% instances that are most likely FG get labelled FG
        idx_fg = idx_fg.nlargest(get_fg, 1, keep='first')
        # Unless the prediction probability is < sl_threshold
        idx_fg = idx_fg[idx_fg[1] > sl_threshold]

        # Track the largest and smallest probability values
        fg_lrg = idx_fg.nlargest(1, 1, keep='first')
        fg_sml = idx_fg.nsmallest(1, 1, keep='first')
        print("LARGEST FG PROB: ", fg_lrg[1])
        print("SMALLEST FG PROB: ", fg_sml[1])


        
        # Take the index of this and add each item to S1_index
        S1_index.extend(idx_nfg.index)
        S1_index.extend(idx_fg.index)
        
        print("WRITING PREDICTED LABELS TO SELF PRED ", loop_counter)
        # Write the predicted label to the relevant location in D
        D['self_pred'].iloc[idx_nfg.index] = 0
        D['self_pred'].iloc[idx_fg.index] = 1
        
        print("==========================================================")
        print(idx_nfg.shape[0], "instances classififed as notFG")
        print(idx_fg.shape[0], "instances classififed as FG")
        print("==========================================================")
        
        # Remove the items from U
        for k in idx_nfg.index:
            U.remove(k)
        
        for k in idx_fg.index:
            U.remove(k)
        
        # train test        
        U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, y_pred_dec_list = \
        trainTest_functions.train_Test_EGAL_SL(chroma_X_df,
                                           S1_index, 
                                           D, 
                                           U, 
                                           L, 
                                           loop_counter, 
                                           y_true_list,
                                           y_pred_list,
                                           trainAcc,
                                           testAcc,
                                           labelled_instances,
                                           y_pred_dec_list)
        
        if len(U) == 0:
            stop_all = True
        
    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, diversity_values_U_L, eucl_distanceS_U_L, CS,\
        CS_sort, CS_sort_filt, beta_, y_pred_dec_list
    
    print("########## EGAL_SL Loop ENDED #############")


# Use EGAL algorithm to select instances for labelling
def cotraining_AL_loop(loop_counter,
                  stop_all, 
                  U, 
                  S1_index, 
                  all_featureSets,
                  D,
                  L,
                  y_true_list,
                  y_pred_list,
                  trainAcc,
                  testAcc,
                  labelled_instances,
                  density_dict,
                  beta_,
                  beta_old,
                  NoL,
                  w,
                  ft,
                  al,
                  y_pred_dec_list):
    
    # convert U to a list so I can use .remove()
    U = U.tolist()
    
# =============================================================================
#     clf_usc = SVC(kernel='rbf',
#               C=50,
#               gamma=1,
#               probability=True)
# =============================================================================
    
    clf_usc = SVC(kernel='linear',
    C=0.1,
    class_weight='balanced',
    probability=True)

    # while there are instances in diversity_values_U_L and stop_all is True
    while len(U) > 0 and not stop_all: 
        # previously used len(S1_index) < D.shape[0] + 1
        # previously used diversity_values_U_L) > 0
        # in place of diversity_values_U_L

        print("EGAL: ", loop_counter, " ::: Feature Set: ", ft, " ::: Alpha Method: ", al)
        print("--------------------------------------------------------------")

        # update diversity values
        diversity_values_U_L, eucl_distanceS_U_L, CS, CS_sort,\
        CS_sort_filt, beta_, stop_all, loop_counter =\
        find_candidate_set_vII(all_featureSets[ft].iloc[S1_index], 
                               all_featureSets[ft].iloc[U], 
                               density_dict, 
                               U,
                               beta_,
                               beta_old,
                               NoL,
                               w, 
                               stop_all,
                               loop_counter)

        print("Size CS_sort_filt = ", CS_sort_filt.shape)
        print("Size diversity_values_U_L = ", diversity_values_U_L.shape)
        print("Number of unlabelled instances, U = ", len(U)) #- in train_Test()
        
        # update S1
        S1_index.extend(CS_sort_filt.index)
        
        # remove these instances from U - wasn't working in train_Test()
        for j in CS_sort_filt.index:
            # delete the element by value NOT index
            U.remove(j)
        
        if stop_all:
            loop_counter = 500
            print("Loop Counter Flipped to ", loop_counter)
        
# =============================================================================
#         # train test        
#         U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
#         labelled_instances, y_pred_dec_list = \
#         trainTest_functions.train_Test_EGAL(all_featureSets[ft],
#                                            S1_index, 
#                                            D, 
#                                            U, 
#                                            L, 
#                                            loop_counter, 
#                                            y_true_list,
#                                            y_pred_list,
#                                            trainAcc,
#                                            testAcc,
#                                            labelled_instances,
#                                            y_pred_dec_list)
# =============================================================================
        
        if len(U) > NoL and not stop_all:

            print("USAL ::: ::: ::: ")
            print("--------------------------------------------------------------")

            # get the following:
            # train data from mfcc0 index S1
            X_train = all_featureSets[0].iloc[S1_index]
            # test data from mfcc0 index everything not in S1
            X_test = all_featureSets[0].drop(S1_index, axis=0)
            # train labels from D index S1
            y_train = D['Cat'].iloc[S1_index]
            # test labels from D index everything not in S1            
            y_test = D['Cat'].drop(S1_index, axis=0) 
    
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
    
            # Fit model
            clf_usc.fit(X_train, y_train)
    
            # get class prediction probability scores in a df
            # use the index from X_test as the index here
            usc_proba_df = pd.DataFrame(clf_usc.predict_proba(X_test),
                                        index=X_test.index)
    
            # get standard deviation of each row (smallest sdev indicates
            # greatest confusion) - THIS IS WHAT I NEED TO CHECK
            # add this as a new column 'stdev' in df
            # usc_proba_df['stdev'] = usc_proba_df.std(axis=1)
            
            # SJ queried the use of stdev here - only two measures.
            # It turns out that if I get the absolute value of the difference
            # between the two class prediction probabilities and rank using
            # that it gives me exactly the same instances in exactly the same
            # order
            usc_proba_df['absdiff'] = abs(usc_proba_df[0] - usc_proba_df[1])
    
            # get the NoL smallest values in usc_proba_df['absdiff']
            idx = usc_proba_df.nsmallest(NoL, 'absdiff', keep='first')
    
            # Take the index of this and add each item to S1_index
            S1_index.extend(idx.index)
    
            # Remove the items from U
            for k in idx.index:
                U.remove(k)
            
            print("--------------------------------------------------------------")
            print("Number of labelled instances = ", len(S1_index))
            print("Number of unlabelled instances, U = ", len(U))
            print("--------------------------------------------------------------")
    
        else:
            # Add the remaining items in U to S1_index
            S1_index.extend(U)
    
            # Empty U so the loop will stop
            U[:] = []
    
            stop_all = True

        if loop_counter % 20 == 0:
            
            print("--------------------------------------------------------------")
            print(":::::::::::::::::::::: Getting a score :::::::::::::::::::::::")
            print("--------------------------------------------------------------")
            
            # Provide this index to train_test_random()
            U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
                labelled_instances, y_pred_dec_list =\
                    trainTest_functions.train_Test_random(all_featureSets[ft],
                                                          S1_index,
                                                          D,
                                                          U,
                                                          L,
                                                          loop_counter, 
                                                          y_true_list,
                                                          y_pred_list,
                                                          trainAcc,
                                                          testAcc,
                                                          labelled_instances,
                                                          y_pred_dec_list)
        
        if testAcc[-1] > 0.9:
            print("--------------------------------------------------------------")
            print("Balanced Test Accuracy has reached: ", testAcc)
            print("Stopping Training")
            print("--------------------------------------------------------------")
            
            stop_all = True
        
        loop_counter = loop_counter + 1
        
    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, diversity_values_U_L, eucl_distanceS_U_L, CS,\
        CS_sort, CS_sort_filt, beta_, y_pred_dec_list
    
    print("########## EGAL Loop ENDED #############")