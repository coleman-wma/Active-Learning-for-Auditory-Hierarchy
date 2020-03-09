#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:18:30 2019

@author: billcoleman
"""


'''
##########################################################
########### RANDOM SELECTION FOR LABELLING ###############
##########################################################
'''
import trainTest_functions

from sklearn.svm import SVC # SVM model
import random
import pandas as pd

from skrvm import RVC

# randomly select instances for labelling
def random_loop(loop_counter,
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
                  NoL,
                  y_pred_dec_list,
                  test_data):
    
    # convert U to a list so I can use .remove()
    U = U.tolist()
    
    # while there are instances in diversity_values_U_L and stop_all is True
    while len(U) > 0 and not stop_all: 
        
        print("RANDOM LOOP ", loop_counter)
        
        # randomly pick NoL instances from U        
        # update the training set index S1_index
        
        if len(U) > NoL:
            # get NoL random instances from U
            new_inst = random.sample(U, NoL)
            #print("Random Instances Selected: ", new_inst)
            S1_index.extend(new_inst)
            
            # remove these instances from U - wasn't working in train_Test()
            for j in new_inst:
                # delete the element by value NOT index
                U.remove(j)
            
        else:
            new_inst = U
            S1_index.extend(new_inst)

            # remove these instances from U - wasn't working in train_Test()
            for j in new_inst:
                # delete the element by value NOT index
                U.remove(j)
                
            stop_all = True
        
        print("Number of instances in S1_index now: ", len(S1_index))
        print("Number of instances in U now: ", len(U))

        if stop_all:
            loop_counter = 500
        
# =============================================================================
#         if testAcc[-1] < 0.9 and len(S1_index) < 1500:
# =============================================================================
        
        # if loop_counter % 20 == 0:
            
        print("--------------------------------------------------------------")
        print(":::::::::::::::::::::: Getting a score :::::::::::::::::::::::")
        print("--------------------------------------------------------------")
    
        # Provide this index to train_test_random()
        U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, y_pred_dec_list = \
        trainTest_functions.train_Test_random(chroma_X_df, 
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
#         if stop_all:
#             break
# =============================================================================
        
        loop_counter = loop_counter + 1

    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc, \
    labelled_instances, y_pred_dec_list
    
    print("########## RANDOM Loop ENDED #############")
    
    
'''
##############################################################
########### UNCERTAINTY SAMPLING FOR LABELLING ###############
##############################################################
'''


# Use Uncertainty Sampling to  select instances for labelling
def usc_loop(loop_counter,
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
                  NoL,
                  y_pred_dec_list,
                  test_data):
    
    # convert U to a list so I can use .remove()
    U = U.tolist()
    
    # GOT AN ERROR USING THIS - LAST y_pred_dec list wasn't 3002 instances long
    # while len(S1_index) < chroma_X_df.shape[0]:
    
    # while there are instances in diversity_values_U_L and stop_all is True
    while len(U) > 0 and not stop_all: 
        
        print("USC LOOP ", loop_counter)
        
        # train a model using instances in S1_index      
        # get a probability score for everything in U
        # rank instances i U by certainty score
        # Get labels for the NoL most uncertain instances
        # Add those to S1, subtract from U
        # Train the model again and loop
        
        if len(U) > NoL:
            # train a model using instances in S1_index
            # toggle probability to enable probability estimates

            # Model derived from fitting on LPMS Scaled data
            clf_usc = SVC(kernel='rbf',
                      C=1,
                      gamma=0.01,
                      class_weight='balanced',
                      probability=False)
            
            # exclude test set labels
            al_portion_labels = D['Cat'].drop(test_data.index, axis=0)

            # get the following:
            # train data on labelled instances
            X_train = chroma_X_df.loc[S1_index]  # changed to loc, was chroma_X_df
            # test data on unlabelled instances
            X_test = chroma_X_df.loc[U]  # was chroma_X_df
            # train labels on labelled instances
            y_train = al_portion_labels.loc[S1_index]  # changed to loc - also changed D['Cat']
            # test labels from unlabelled instances         
            y_test = al_portion_labels.loc[U]

            y_train = y_train.astype('int')
            y_test = y_test.astype('int')

            # Fit model
            clf_usc.fit(X_train, y_train)

            # SWITCHING TO USING CONFIDENCE SCORES

            usc_confidence_df = pd.DataFrame(clf_usc.decision_function(X_test),
                index=X_test.index)

            usc_confidence_df['absdiff'] = abs(usc_confidence_df[0])

            # get the NoL smallest values in usc_proba_df['absdiff']
            idx = usc_confidence_df.nsmallest(NoL, 'absdiff', keep='first')
            
            print("Adding ", len(idx.index), "instances to S1_index.")
            if len(S1_index) < 11:
                print("First USAL instances selected are: ", idx.index)

            # Take the index of this and add each item to S1_index
            S1_index.extend(idx.index)
            print("Number of instances in S1_index now: ", len(S1_index))

            # Remove the items from U
            for k in idx.index:
                U.remove(k)
            
            print("Number of instances in U now: ", len(U))

        else:
            # Add the remaining items in U to S1_index
            S1_index.extend(U)
            print("Adding ", len(U), "instances to S1_index.")
            print("Number of instances in S1_index now: ", len(S1_index))

            # Empty U so the loop will stop
            U[:] = []
            print("Number of instances in U now: ", len(U))
            
            stop_all = True
        
        if stop_all:
            loop_counter = 500
        
        # if loop_counter % 20 == 0:
            
        print("--------------------------------------------------------------")
        print(":::::::::::::::::::::: Getting a score :::::::::::::::::::::::")
        print("--------------------------------------------------------------")

        # Provide this index to train_test_random()
        U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
            labelled_instances, y_pred_dec_list =\
                trainTest_functions.train_Test_random(chroma_X_df,
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
        
        loop_counter = loop_counter + 1

    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
    labelled_instances, stop_all, y_pred_dec_list


# Use Uncertainty Sampling to  select instances for labelling
def usc_loop_self(loop_counter,
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
                  NoL,
                  y_pred_dec_list):

    # convert U to a list so I can use .remove()
    U = U.tolist()

    # while there are instances in diversity_values_U_L and stop_all is True
    while len(S1_index) < chroma_X_df.shape[0] and not stop_all:

        print("SELF TRAINING LOOP ", loop_counter)

        # train a model using instances in S1_index      
        # get a probability score for everything in U
        # rank instances in U by certainty score
        # Get labels for the NoL most uncertain instances
        # Add those to S1, subtract from U
        # NoL instances with highest probability get assigned predicted labels
        # Add those to S1, subtract from U
        # Train the model again and loop
        
        u_change = len(U)

        if len(U) > NoL:
            
            # These are the model params used elsewhere - confusion over
            # which runs have which model as the one just below was also used
            clf_usc = SVC(kernel='rbf',
                          C=50,
                          gamma=1,
                          probability=True)

# =============================================================================
#             # Base model used in initial investigations
#             clf_usc = SVC(gamma='scale',
#                           probability=True,
#                           class_weight='balanced')
# =============================================================================

            # get the following:
            # train data from chroma_X_df index S1
            X_train = chroma_X_df.iloc[S1_index]
            # test data from chroma_X_df index everything not in S1
            X_test = chroma_X_df.drop(S1_index, axis=0)
            # train labels from D index S1
            y_train = D['self_pred'].iloc[S1_index]
            # test labels from D index everything not in S1            
            y_test = D['self_pred'].drop(S1_index, axis=0) 
            
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
            
            # Fit model
            clf_usc.fit(X_train, y_train)
            
            # get class prediction probability scores in a df
            # use the index from X_test as the index here
            usc_proba_df = pd.DataFrame(clf_usc.predict_proba(X_test),
                                        index=X_test.index)
            
# =============================================================================
#             # Get the difference between the two probabilities
#             usc_proba_df['absdiff'] = abs(usc_proba_df[0] - usc_proba_df[1])
# 
#             ### Problem here is everything is getting labeled nonFG
#             # get the NoL largest values in usc_proba_df['absdiff']
#             # these will be the instances the model is most certain of
#             idx = usc_proba_df.nlargest(NoL, 'absdiff', keep='first')
#             
#             # Columns '0' for notFG and '1' for FG
#             # Anything that scores over 0.9 in that category is given that label
#             # Scores that are tied are not allocated so go around again
#             # Tried 0.5 as the threshold here as well with v. similar result
#             idx_nfg = idx[idx[0] >= 0.9]
#             idx_fg = idx[idx[1] >= 0.9]
# =============================================================================

            # Try to force more FG labelling
            # variable schema worked best on mfcc0
            # thresh = 0.5
            
# =============================================================================
#             # Trying a variable threshold
#             if len(S1_index) < 750:
#                 thresh = 0.48
#             elif len(S1_index) < 1150:
#                 thresh = 0.55
#             elif len(S1_index) < 1550:
#                 thresh = 0.6
#             else:
#                 thresh = 0.7
# =============================================================================
            
            # print("THRESHOLD = ", thresh)
            get_fg = round(NoL * 0.2)
            get_nfg = NoL - get_fg
            
            # This will have to be tweaked for each dataset probably
            # below works well on mfcc0 - but needs to be controlled so
            # an unlimited number of FG labels aren't taken
# =============================================================================
#             idx_fg = usc_proba_df[usc_proba_df[1] > thresh]
#             idx_fg = idx_fg.nlargest(NoL, 1, keep='first')
# =============================================================================
            
            idx_fg = usc_proba_df.nlargest(get_fg, 1, keep='first')
            idx_fg = idx_fg[idx_fg[1] > 0.45]

            fg_lrg = idx_fg.nlargest(1, 1, keep='first')
            fg_sml = idx_fg.nsmallest(1, 1, keep='first')
            
            print("LARGEST FG PROB: ", fg_lrg[1])
            print("SMALLEST FG PROB: ", fg_sml[1])
            
            # to keep loop at same number, get diff btw NoL and the no. of FG labels
            # get = NoL - idx_fg.shape[0]
            
            # Get this number of nonFG labels
# =============================================================================
#             idx_nfg = usc_proba_df[usc_proba_df[0] > thresh]
#             idx_nfg = idx_nfg.nlargest(get, 0, keep='first')
# =============================================================================
            
            idx_nfg = usc_proba_df.nlargest(get_nfg, 0, keep='first')
            idx_nfg = idx_nfg[idx_nfg[0] > 0.45]

            nfg_lrg = idx_nfg.nlargest(1, 0, keep='first')
            nfg_sml = idx_nfg.nsmallest(1, 0, keep='first')
            
            print("LARGEST nFG PROB: ", nfg_lrg[0])
            print("SMALLEST nFG PROB: ", nfg_sml[0])
            
            # Take the index of this and add each item to S1_index
            S1_index.extend(idx_nfg.index)
            S1_index.extend(idx_fg.index)
            
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
            
            if len(U) == u_change:
                print("Labels predicted for all instances to confidence threshold 0.9")
                print("STOPPING SELF LEARNING AT U SIZE: ", len(U))
                stop_all = True
            
        else:

            # These are the model params used elsewhere - confusion over
            # which runs have which model as the one just below was also used
            clf_usc = SVC(kernel='rbf',
                          C=50,
                          gamma=1,
                          probability=True)
            
# =============================================================================
#             # Base model used in initial investigations
#             clf_usc = SVC(gamma='scale',
#                           probability=True,
#                           class_weight='balanced')
# =============================================================================

            # get the following:
            # train data from chroma_X_df index S1
            X_train = chroma_X_df.iloc[S1_index]
            # test data from chroma_X_df index everything not in S1
            X_test = chroma_X_df.drop(S1_index, axis=0)
            # train labels from D index S1
            y_train = D['self_pred'].iloc[S1_index]
            # test labels from D index everything not in S1            
            y_test = D['self_pred'].drop(S1_index, axis=0) 
            
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
            
            # Fit model
            clf_usc.fit(X_train, y_train)
            
            # get class prediction probability scores in a df
            # use the index from X_test as the index here
            usc_proba_df = pd.DataFrame(clf_usc.predict_proba(X_test),
                                        index=X_test.index)
            
            # Columns '0' for notFG and '1' for FG
            # Anything that scores over 0.5 in that category is given that label
            # This is the last loop so just split on 0.5
            
            idx_nfg = usc_proba_df[usc_proba_df[0] > 0.5]
            idx_fg = usc_proba_df[usc_proba_df[1] >= 0.5]

            # Take the index of this and add each item to S1_index
            S1_index.extend(idx_nfg.index)
            S1_index.extend(idx_fg.index)
            
            # Write the predicted label to the relevant location in D
            D['self_pred'].iloc[idx_nfg.index] = 0
            D['self_pred'].iloc[idx_fg.index] = 1
            
            print(idx_nfg.shape[0], "instances classififed as notFG")
            print(idx_fg.shape[0], "instances classififed as FG")
            
            # Remove the items from U
            for k in idx_nfg.index:
                U.remove(k)
            
            for k in idx_fg.index:
                U.remove(k)
            
            stop_all = True
            
            
        if stop_all:
            loop_counter = 500
            
            
        # Provide this index to train_test_random()
        U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
            labelled_instances, y_pred_dec_list =\
                trainTest_functions.train_Test_self_II(chroma_X_df,
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
        
        if stop_all:
            print("STOP ALL")
            break
        
    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc, \
    labelled_instances, stop_all, D, y_pred_dec_list
        
    print("########## SELF TRAINING Loop ENDED #############")


# Use Uncertainty Sampling and Self Learning to  select instances for labelling
def USAL_SL_loop(loop_counter,
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
                  NoL,
                  y_pred_dec_list,
                  sl_threshold):

    # convert U to a list so I can use .remove()
    U = U.tolist()

    # while there are instances in diversity_values_U_L and stop_all is True
    while len(S1_index) < chroma_X_df.shape[0] and not stop_all:

        print("USAL_SL TRAINING LOOP ", loop_counter)

        # train a model using instances in S1_index      
        # get a probability score for everything in U
        # rank instances in U by certainty score
        # Get labels for the NoL most uncertain instances
        # Add those to S1, subtract from U
        # NoL instances with highest probability get assigned predicted labels
        # Add those to S1, subtract from U
        # Train the model again
        # Get the NoL instances the model is most confident about
        # Apply predicted labels to those instances
        # Feed them into training index S1_index
        # Train the model and predict labels for all dataset
        # Compare to actual labels to score
        
        # These are the model params used elsewhere - confusion over
        # which runs have which model as the one just below was also used
        # model with rbf kernel has been used mostly for balAcc calculations
        # as all other models were very erratic
# =============================================================================
#         clf_usc = SVC(kernel='rbf',
#                       C=50,
#                       gamma=1,
#                       probability=True)
# =============================================================================
        
        # Base model used in initial investigations
        clf_usc = SVC(gamma='scale',
                      probability=True,
                      class_weight='balanced')
        
        u_change = len(U)

        if len(U) > NoL:

            # These are the model params used elsewhere - confusion over
            # which runs have which model as the one just below was also used
# =============================================================================
#             clf_usc = SVC(kernel='rbf',
#                           C=50,
#                           gamma=1,
#                           probability=True)
# =============================================================================
            
# =============================================================================
#             # Base model used in initial investigations
#             clf_usc = SVC(gamma='scale',
#                           probability=True,
#                           class_weight='balanced')
# =============================================================================
            
                        # get the following:
            # train data from chroma_X_df index S1
            X_train = chroma_X_df.iloc[S1_index]
            # test data from chroma_X_df index everything not in S1
            X_test = chroma_X_df.drop(S1_index, axis=0)
            # train labels from D index S1
            y_train = D['self_pred'].iloc[S1_index]
            # test labels from D index everything not in S1            
            # y_test = D['self_pred'].drop(S1_index, axis=0)

            y_train = y_train.astype('int')
            # y_test = y_test.astype('int')

            # Fit model
            clf_usc.fit(X_train, y_train)

            # get class prediction probability scores in a df
            # use the index from X_test as the index here
            usc_proba_df = pd.DataFrame(clf_usc.predict_proba(X_test),
                                        index=X_test.index)

            # get the absolute value of the difference
            # between the two class prediction probabilities and rank
            usc_proba_df['absdiff'] = abs(usc_proba_df[0] - usc_proba_df[1])

            print("IDing ACTUAL LABELS TO GET ", loop_counter)
            # get the NoL smallest values in usc_proba_df['absdiff']
            idx = usc_proba_df.nsmallest(NoL, 'absdiff', keep='first')

            # Take the index of this and add each item to S1_index
            S1_index.extend(idx.index)

            # Remove the items from U
            for k in idx.index:
                U.remove(k)
            
            # Provide this index to train_test_random()
            U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
                labelled_instances, y_pred_dec_list =\
                trainTest_functions.train_Test_self_II(chroma_X_df,
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
            
            #############################################

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
            clf_usc.fit(X_train_, y_train_)
            
            # get class prediction probability scores in a df
            # use the index from X_test as the index here
            usc_proba_df_ = pd.DataFrame(clf_usc.predict_proba(X_test_),
                                        index=X_test_.index)
            
            # limit the number of instances that can be tagged as FG to 20%
            get_fg = round(NoL * 0.2)
            get_nfg = NoL - get_fg
            
            
# =============================================================================
#             # 20% instances that are most likely FG get labelled FG
#             idx_fg = usc_proba_df_.nlargest(get_fg, 1, keep='first')
#             # Unless the prediction probability is < 45%
#             idx_fg = idx_fg[idx_fg[1] > 0.45]
# 
#             # Track the largest and smallest probability values
#             fg_lrg = idx_fg.nlargest(1, 1, keep='first')
#             fg_sml = idx_fg.nsmallest(1, 1, keep='first')
#             print("LARGEST FG PROB: ", fg_lrg[1])
#             print("SMALLEST FG PROB: ", fg_sml[1])
# 
#             # 80% instances that are most likely nonFG get labelled nonFG         
#             idx_nfg = usc_proba_df_.nlargest(get_nfg, 0, keep='first')
#             # Unless the prediction probability is < 45%
#             idx_nfg = idx_nfg[idx_nfg[0] > 0.45]
# 
#             # Track the largest and smallest probability values
#             nfg_lrg = idx_nfg.nlargest(1, 0, keep='first')
#             nfg_sml = idx_nfg.nsmallest(1, 0, keep='first')
#             print("LARGEST nFG PROB: ", nfg_lrg[0])
#             print("SMALLEST nFG PROB: ", nfg_sml[0])
# =============================================================================
            
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
            
            if len(U) == u_change:
                print("Labels predicted for all instances to confidence threshold 0.9")
                print("STOPPING SELF LEARNING AT U SIZE: ", len(U))
                stop_all = True
            
            # Provide this index to train_test_random()
            U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
                labelled_instances, y_pred_dec_list =\
                trainTest_functions.train_Test_self_II(chroma_X_df,
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
            
        else:

# =============================================================================
#             # These are the model params used elsewhere - confusion over
#             # which runs have which model as the one just below was also used
#             clf_usc = SVC(kernel='rbf',
#                           C=50,
#                           gamma=1,
#                           probability=True)
# =============================================================================
            
# =============================================================================
#             # Base model used in initial investigations
#             clf_usc = SVC(gamma='scale',
#                           probability=True,
#                           class_weight='balanced')
# =============================================================================

            # get the following:
            # train data from chroma_X_df index S1
            X_train = chroma_X_df.iloc[S1_index]
            # test data from chroma_X_df index everything not in S1
            X_test = chroma_X_df.drop(S1_index, axis=0)
            # train labels from D index S1
            y_train = D['self_pred'].iloc[S1_index]
            # test labels from D index everything not in S1            
            y_test = D['self_pred'].drop(S1_index, axis=0) 
            
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
            
            # Fit model
            clf_usc.fit(X_train, y_train)
            
            # get class prediction probability scores in a df
            # use the index from X_test as the index here
            usc_proba_df = pd.DataFrame(clf_usc.predict_proba(X_test),
                                        index=X_test.index)
            
            # get the absolute value of the difference
            # between the two class prediction probabilities and rank
            usc_proba_df['absdiff'] = abs(usc_proba_df[0] - usc_proba_df[1])

            print("FINALFINAL - IDing ACTUAL LABELS TO GET ", loop_counter)
            # get the NoL smallest values in usc_proba_df['absdiff']
            idx = usc_proba_df.nsmallest(NoL, 'absdiff', keep='first')

            # Take the index of this and add each item to S1_index
            S1_index.extend(idx.index)

            # Remove the items from U
            for k in idx.index:
                U.remove(k)
            
            # Provide this index to train_test_random()
            U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
                labelled_instances, y_pred_dec_list =\
                trainTest_functions.train_Test_self_II(chroma_X_df,
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

            stop_all = True
            
        if stop_all:
            loop_counter = 500
        
        if stop_all:
            print("STOP ALL")
            break
        
    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc, \
    labelled_instances, stop_all, D, y_pred_dec_list
        
    print("########## SELF TRAINING Loop ENDED #############")


# Use Uncertainty Sampling to  select instances for labelling
def self_training_loop(loop_counter,
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
                  NoL,
                  y_pred_dec_list):

    # convert U to a list so I can use .remove()
    U = U.tolist()

    # while there are instances in diversity_values_U_L and stop_all is True
    while len(S1_index) < chroma_X_df.shape[0] and not stop_all:

        print("SELF TRAINING LOOP ", loop_counter)

        # train a model using instances in S1_index      
        # get a probability score for everything in U
        # rank instances in U by certainty score
        # NoL instances with highest probability get assigned predicted labels
        # Add those to S1, subtract from U
        # Train the model again and loop
        
        u_change = len(U)

        if len(U) > NoL:
            
            # These are the model params used elsewhere
            clf_usc = SVC(kernel='rbf',
                          C=50,
                          gamma=1,
                          probability=True)

            # get the following:
            # train data from chroma_X_df index S1
            X_train = chroma_X_df.iloc[S1_index]
            # test data from chroma_X_df index everything not in S1
            X_test = chroma_X_df.drop(S1_index, axis=0)
            # train labels from D index S1
            y_train = D['self_pred'].iloc[S1_index]
            # test labels from D index everything not in S1            
            y_test = D['self_pred'].drop(S1_index, axis=0) 
            
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
            
            # Fit model
            clf_usc.fit(X_train, y_train)
            
            # get class prediction probability scores in a df
            # use the index from X_test as the index here
            usc_proba_df = pd.DataFrame(clf_usc.predict_proba(X_test),
                                        index=X_test.index)
            
            # set a target of FG and nonFG labels to predict each round
            get_fg = round(NoL * 0.2)
            get_nfg = NoL - get_fg
            
            # get the indices of the most certain FG predictions
            idx_fg = usc_proba_df.nlargest(get_fg, 1, keep='first')
            # if probability is no higher than 45% don't take them though
            # idx_fg = idx_fg[idx_fg[1] > 0.45]

            # track what the largest and smallest probabilities are
            fg_lrg = idx_fg.nlargest(1, 1, keep='first')
            fg_sml = idx_fg.nsmallest(1, 1, keep='first')
            print("LARGEST FG PROB: ", fg_lrg[1])
            print("SMALLEST FG PROB: ", fg_sml[1])
            
            # Write the predicted label to the relevant location in D
            D['self_pred'].iloc[idx_fg.index] = 1
            # Remove these instances from U
            for k in idx_fg.index:
                U.remove(k)
            # Remove them also from the dataframe (making a copy here, same
            # effect though)
            mod_usc_proba_df = usc_proba_df.drop(idx_fg.index)

            # Do the same for nonFG
            # get the inddices of the most confident nonFG probabilities            
            idx_nfg = mod_usc_proba_df.nlargest(get_nfg, 0, keep='first')
            # If probability is no higher than 45% discard them though
            idx_nfg = idx_nfg[idx_nfg[0] > 0.45]

            # Track what smallest and largest probabilities are
            nfg_lrg = idx_nfg.nlargest(1, 0, keep='first')
            nfg_sml = idx_nfg.nsmallest(1, 0, keep='first')
            print("LARGEST nFG PROB: ", nfg_lrg[0])
            print("SMALLEST nFG PROB: ", nfg_sml[0])
            
            # Take the index of this and add each item to S1_index
            S1_index.extend(idx_nfg.index)
            S1_index.extend(idx_fg.index)
            
            # Write the predicted label to the relevant location in D
            D['self_pred'].iloc[idx_nfg.index] = 0
            
            
            print("==========================================================")
            print(idx_nfg.shape[0], "instances classified as notFG")
            print(idx_fg.shape[0], "instances classified as FG")
            print("==========================================================")
            
            # Remove the items from U
            for k in idx_nfg.index:
                U.remove(k)
            
            if len(U) == u_change:
                print("NO LABELS PREDICTED!!")
                print("STOPPING SELF LEARNING AT U SIZE: ", len(U))
                stop_all = True
            
        else:

            # These are the model params used elsewhere - confusion over
            # which runs have which model as the one just below was also used
            clf_usc = SVC(kernel='rbf',
                          C=50,
                          gamma=1,
                          probability=True)

            # get the following:
            # train data from chroma_X_df index S1
            X_train = chroma_X_df.iloc[S1_index]
            # test data from chroma_X_df index everything not in S1
            X_test = chroma_X_df.drop(S1_index, axis=0)
            # train labels from D index S1
            y_train = D['self_pred'].iloc[S1_index]
            # test labels from D index everything not in S1            
            y_test = D['self_pred'].drop(S1_index, axis=0) 
            
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
            
            # Fit model
            clf_usc.fit(X_train, y_train)
            
            # get class prediction probability scores in a df
            # use the index from X_test as the index here
            usc_proba_df = pd.DataFrame(clf_usc.predict_proba(X_test),
                                        index=X_test.index)
            
            # Columns '0' for notFG and '1' for FG
            # Anything that scores over 0.5 in that category is given that label
            # This is the last loop so just split on 0.5
            
            idx_nfg = usc_proba_df[usc_proba_df[0] > 0.5]
            idx_fg = usc_proba_df[usc_proba_df[1] >= 0.5]

            # Take the index of this and add each item to S1_index
            S1_index.extend(idx_nfg.index)
            S1_index.extend(idx_fg.index)
            
            # Write the predicted label to the relevant location in D
            D['self_pred'].iloc[idx_nfg.index] = 0
            D['self_pred'].iloc[idx_fg.index] = 1
            
            print(idx_nfg.shape[0], "instances classififed as notFG")
            print(idx_fg.shape[0], "instances classififed as FG")
            
            # Remove the items from U
            for k in idx_nfg.index:
                U.remove(k)
            
            for k in idx_fg.index:
                U.remove(k)
            
            stop_all = True
            
            
        if stop_all:
            loop_counter = 500
            
        print("--------------------------------------------------------------")
        print("::: LENGTH OF S1_index:, ", len(S1_index), "::::::::::::::::::")
        print(":::::::::: LENGTH OF U:, ", len(U), ":::::::::::::::::::::::::")
        print("--------------------------------------------------------------")
        
        if testAcc[-1] < 0.9 and len(S1_index) < 1500:
        
            if loop_counter % 20 == 0:
                
                print("--------------------------------------------------------------")
                print(":::::::::::::::::::::: Getting a score :::::::::::::::::::::::")
                print("--------------------------------------------------------------")
    
                # Provide this index to train_test_random()
                U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
                    labelled_instances, y_pred_dec_list =\
                        trainTest_functions.train_Test_self_II(chroma_X_df,
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
#                 if testAcc[-1] < 0.7 and len(S1_index) > 800:
#                     # then this run isn't preforming well at all
#                     # set NoL=500 to expedite run
#                     print("This run is performing poorly, so lets EXPEDITE. NoL=500")
#                     NoL=500
#                 
#                 if testAcc[-1] < 0.55 and len(S1_index) > 300:
#                     # then this run isn't preforming well at all
#                     # set NoL=500 to expedite run
#                     print("This run is performing poorly, so lets EXPEDITE. NoL=500")
#                     NoL=500
# =============================================================================
        
        else:
            print("Score now above 0.9 or len(S1_index) > 1500, EXPEDITE. NoL=500")
            NoL = 500
            
            print("--------------------------------------------------------------")
            print(":::::::::::::::::::::: Getting a score :::::::::::::::::::::::")
            print("--------------------------------------------------------------")
    
            # Provide this index to train_test_random()
            U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
                labelled_instances, y_pred_dec_list =\
                trainTest_functions.train_Test_self_II(chroma_X_df,
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
        
        if stop_all:
            print("STOP ALL")
            break
        
        loop_counter = loop_counter + 1
        
    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
    labelled_instances, stop_all, D, y_pred_dec_list
        
    print("########## SELF TRAINING Loop ENDED #############")