#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:16:24 2019

@author: billcoleman
"""


'''
############################################
######### TRAIN/TEST FUNCTIONS #############
############################################

# going to be changing train and test each time
# take index from S1_index, build X_train, X_test, y_train, y_test
# training data is everything in S1
# test data is everything not in S1

FROM HERE, S1 CAN BE USED AS THE CONTROLLER OF WHAT'S IN THE LABELLED POOL OR NOT

'''

from sklearn.svm import SVC # SVM model
from sklearn.metrics import classification_report, balanced_accuracy_score

# train and test a SVM model on sets controlled by the contents of S1_index
def train_Test(chroma_X_df, S1_index, D, U, L, loop_counter, y_true_list,
               y_pred_list, trainAcc, testAcc, labelled_instances,
               y_pred_dec_list, test_data):

    # get the following:
    # train data from chroma_X_df index S1
    X_train = chroma_X_df.loc[S1_index]  # changed to loc
    # test data from chroma_X_df index everything
    X_test = test_data
    # train labels from D index S1
    y_train = D['Cat'].iloc[S1_index]
    # test labels from D test data
    y_test = D['Cat'].iloc[test_data.index]
    
    make_U = chroma_X_df.drop(S1_index, axis=0)
    
    # used in find_candidate_set() & get_sorted_density_values()
    U = make_U.index.values
    # this is basically S1_index - used in find_candidate_set()
    L = X_train.index.values
    
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    
    print("SHAPES: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("Number of labelled instances, S1_index = ", len(S1_index))
    print("Number of unlabelled instances, U = ", len(U))

    # Model derived from fitting on LPMS Scaled data
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
    
    # For AUC calculation
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)
    
    # this is a python thing - I think if something is created in a global
    # scope it gettings tracked globally, so anytime there's a reference to it
    # every location gets updated. This was appending all slots with the
    # latest iteration of S1_index - the syntax below means every slot has the
    # corresponding S1_index for that Loop. I think the y_true and y_pred are
    # different because they're local variables, or updated locally.
    y_pred_dec_list.append(S1_index.copy())

    labelled_instances.append(X_train.shape[0])
    testAcc.append(bal_acc)

    # increment loop counter
    # loop_counter = loop_counter + 1

    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, y_pred_dec_list


# train and test a SVM model on sets controlled by the contents of S1_index
def train_Test_random(chroma_X_df, S1_index, D, U, L, loop_counter,
                      y_true_list, y_pred_list, trainAcc, testAcc,
                      labelled_instances, y_pred_dec_list, test_data):
    
    # get the following:
    # train data from chroma_X_df index S1
    X_train = chroma_X_df.loc[S1_index]   # changed to loc
    # test data from chroma_X_df index everything not in S1
    X_test = test_data
    # train labels from D index S1
    y_train = D['Cat'].loc[S1_index]    # changed to loc
    # test labels from D index test data
    y_test = D['Cat'].iloc[test_data.index]
    
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    
    print("SHAPES: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("Number of labelled instances, S1_index = ", len(S1_index))
    print("Number of unlabelled instances, U = ", len(U))
    
    # Model derived from fitting on LPMS Scaled data
    clf = SVC(kernel='rbf',
              C=1,
              gamma=0.01,
              class_weight='balanced')

    # Fit model
    clf.fit(X_train, y_train)
    
    # BALANCED accuracy scores
    # y_pred_dec = clf.decision_function(X_test)
    y_true, y_pred = y_test, clf.predict(X_test) 
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print("------------------------------------------------------------------")
    print('Balanced accuracy on test set (y_true Vs y_pred): ', bal_acc)
    print("------------------------------------------------------------------")
    
    # For AUC calculation
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)

    # this is a python thing - I think if something is created in a global
    # scope it gettings tracked globally, so anytime there's a reference to it
    # every location gets updated. This was appending all slots with the
    # latest iteration of S1_index - the syntax below means every slot has the
    # corresponding S1_index for that Loop. I think the y_true and y_pred are
    # different because they're local variables, or updated locally.
    y_pred_dec_list.append(S1_index.copy())

    labelled_instances.append(X_train.shape[0])
    testAcc.append(bal_acc)

    # increment loop counter
    # loop_counter = loop_counter + 1

    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, y_pred_dec_list



# train and test a SVM model on sets controlled by the contents of S1_index
def train_Test_EGAL(chroma_X_df, S1_index, D, U, L, loop_counter, y_true_list, 
                   y_pred_list, trainAcc, testAcc, labelled_instances,
                   y_pred_dec_list, test_data):
    
    # get the following:
    # train data from chroma_X_df index S1
    X_train = chroma_X_df.loc[S1_index]
    # test data from chroma_X_df index everything not in S1
    X_test = test_data
    # train labels from D index S1
    y_train = D['Cat'].iloc[S1_index]
    # test labels from D index everything not in S1
    y_test = D['Cat'].iloc[test_data.index]
    
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    print("SHAPES: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("Number of labelled instances, S1_index = ", len(S1_index))
    print("Number of unlabelled instances, U = ", len(U))
    
    # Model derived from fitting on LPMS Scaled data
    clf = SVC(kernel='rbf',
              C=1,
              gamma=0.01,
              class_weight='balanced')
    
# =============================================================================
#     # Base model used in initial investigations
#     clf = SVC(kernel='rbf',
#               C=50,
#               gamma=1,
#               class_weight='balanced')
# =============================================================================
    
# =============================================================================
#     # seems to work better on LPMS??
#     clf = SVC(kernel='linear',
#           C=0.1,
#           class_weight='balanced')
# =============================================================================
    
# =============================================================================
#     clf = SVC(kernel='poly',
#           C=1,
#           gamma=1,
#           class_weight='balanced')
# =============================================================================
    
    # best model according to parameter search on mfcc0 data
    # clf = SVC(kernel='rbf', C=0.1, gamma=0.01, class_weight='balanced')
    
    # best model according to parameter grid search - chroma delta feature
    # clf = SVC(kernel='poly', C=1, gamma=10, class_weight='balanced')

    # Fit model
    clf.fit(X_train, y_train) # .values.ravel()

    # BALANCED accuracy scores
    y_true, y_pred = y_test, clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print("------------------------------------------------------------------")
    print('Balanced accuracy on test set (y_true Vs y_pred): ', bal_acc)
    print("------------------------------------------------------------------")

    # For AUC calculation
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)
    
    # this is a python thing - I think if something is created in a global
    # scope it gettings tracked globally, so anytime there's a reference to it
    # every location gets updated. This was appending all slots with the
    # latest iteration of S1_index - the syntax below means every slot has the
    # corresponding S1_index for that Loop. I think the y_true and y_pred are
    # different because they're local variables, or updated locally.
    y_pred_dec_list.append(S1_index.copy())

    labelled_instances.append(X_train.shape[0])
    testAcc.append(bal_acc)

    # increment loop counter
    # loop_counter = loop_counter + 1

    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, y_pred_dec_list


# train and test a SVM model on sets controlled by the contents of S1_index
def train_Test_EGAL_SL(chroma_X_df, S1_index, D, U, L, loop_counter, y_true_list, 
                   y_pred_list, trainAcc, testAcc, labelled_instances,
                   y_pred_dec_list):
    
    # get the following:
    # train data from chroma_X_df index S1
    X_train = chroma_X_df.iloc[S1_index]
    # test data from chroma_X_df index everything not in S1
    X_test = chroma_X_df
    # train labels from D index S1
    y_train = D['self_pred'].iloc[S1_index]
    # test labels from D index everything not in S1
    y_test = D['Cat']
    
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    print("SHAPES: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("Number of labelled instances, S1_index = ", len(S1_index))
    print("Number of unlabelled instances, U = ", len(U))
    
    # Base model used in initial investigations
    clf = SVC(kernel='rbf',
              C=50,
              gamma=1,
              class_weight='balanced')
    
    # best model according to parameter search on mfcc0 data
    # clf = SVC(kernel='rbf', C=0.1, gamma=0.01, class_weight='balanced')
    
    # best model according to parameter grid search - chroma delta feature
    # clf = SVC(kernel='poly', C=1, gamma=10, class_weight='balanced')

    # Fit model
    clf.fit(X_train, y_train) # .values.ravel()

    # BALANCED accuracy scores
    y_true, y_pred = y_test, clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print('Balanced accuracy on test set (y_true Vs y_pred: ', bal_acc)
    print("------------------------------------------------------------------")

    # For AUC calculation
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)
    
    # this is a python thing - I think if something is created in a global
    # scope it gettings tracked globally, so anytime there's a reference to it
    # every location gets updated. This was appending all slots with the
    # latest iteration of S1_index - the syntax below means every slot has the
    # corresponding S1_index for that Loop. I think the y_true and y_pred are
    # different because they're local variables, or updated locally.
    y_pred_dec_list.append(S1_index.copy())

    labelled_instances.append(X_train.shape[0])
    testAcc.append(bal_acc)

    # increment loop counter
    loop_counter = loop_counter + 1

    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, y_pred_dec_list


# train and test a SVM model on sets controlled by the contents of S1_index
# Add a self-learning element
def train_Test_self_I(chroma_X_df, S1_index, D, U, L, loop_counter,
                      y_true_list, y_pred_list, trainAcc, testAcc,
                      labelled_instances, y_pred_dec_list):

    # get the following:
    # train data from chroma_X_df index S1
    X_train = chroma_X_df.iloc[S1_index]
    # test data from chroma_X_df index everything not in S1
    X_test = chroma_X_df
    # train labels from D index S1
    y_train = D['self_pred'].iloc[S1_index]
    # test labels from D index everything not in S1
    y_test = D['Cat']
    
    make_U = chroma_X_df.drop(S1_index, axis=0)
    
    # used in find_candidate_set() & get_sorted_density_values()
    U = make_U.index.values
    # this is basically S1_index - used in find_candidate_set()
    L = X_train.index.values

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    print("SHAPES: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("Number of labelled instances, S1_index = ", len(S1_index))
    print("Number of unlabelled instances, U = ", len(U))

    # Base model used in initial investigations
    clf = SVC(kernel='rbf',
              C=50,
              gamma=1,
              class_weight='balanced')

    # Fit model
    clf.fit(X_train, y_train)
    
    # BALANCED accuracy scores
    y_true, y_pred = y_test, clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print('Balanced accuracy on test set (y_true Vs y_pred: ', bal_acc)
    print("------------------------------------------------------------------")
    
    # For AUC calculation
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)
    
    # this is a python thing - I think if something is created in a global
    # scope it gettings tracked globally, so anytime there's a reference to it
    # every location gets updated. This was appending all slots with the
    # latest iteration of S1_index - the syntax below means every slot has the
    # corresponding S1_index for that Loop. I think the y_true and y_pred are
    # different because they're local variables, or updated locally.
    y_pred_dec_list.append(S1_index.copy())

    labelled_instances.append(X_train.shape[0])
    testAcc.append(bal_acc)
        
    # increment loop counter
    loop_counter = loop_counter + 1
    
    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, y_pred_dec_list


# train and test a SVM model on sets controlled by the contents of S1_index
def train_Test_self_II(chroma_X_df, S1_index, D, U, L, loop_counter,
                      y_true_list, y_pred_list, trainAcc, testAcc,
                      labelled_instances, y_pred_dec_list):
    
    # get the following:
    # train data from chroma_X_df index S1
    X_train = chroma_X_df.iloc[S1_index]
    # test data from chroma_X_df index everything not in S1
    X_test = chroma_X_df
    # train labels from D index S1
    y_train = D['self_pred'].iloc[S1_index]
    # test labels from D index everything not in S1
    y_test = D['Cat']
    
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    
    print("SHAPES: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("Number of labelled instances, S1_index = ", len(S1_index))
    print("Number of unlabelled instances, U = ", len(U))

    # Base model used in initial investigations
    clf = SVC(kernel='rbf',
              C=50,
              gamma=1,
              class_weight='balanced')
    
    # best model according to parameter grid search - chroma delta feature
    # clf = SVC(kernel='poly', C=1, gamma=10, class_weight='balanced')

    # Fit model
    clf.fit(X_train, y_train)
    
    # BALANCED accuracy scores
    y_true, y_pred = y_test, clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print('Balanced accuracy on test set (y_true Vs y_pred): ', bal_acc)
    print("------------------------------------------------------------------")
    
    # For AUC calculation
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)
    
    # this is a python thing - I think if something is created in a global
    # scope it gettings tracked globally, so anytime there's a reference to it
    # every location gets updated. This was appending all slots with the
    # latest iteration of S1_index - the syntax below means every slot has the
    # corresponding S1_index for that Loop. I think the y_true and y_pred are
    # different because they're local variables, or updated locally.
    y_pred_dec_list.append(S1_index.copy())

    labelled_instances.append(X_train.shape[0])
    testAcc.append(bal_acc)

    # increment loop counter
    # loop_counter = loop_counter + 1

    return U, L, loop_counter, y_true_list, y_pred_list, trainAcc, testAcc,\
        labelled_instances, y_pred_dec_list


# train and test a SVM model on sets controlled by the contents of S1_index
def diff_models(chroma_X_df, S1_index, D, U, L, loop_counter, y_true_list,
               y_pred_list, trainAcc, testAcc, labelled_instances):

    # get the following:
    # train data from chroma_X_df index S1
    X_train = chroma_X_df.iloc[S1_index]
    # test data from chroma_X_df index everything
    X_test = chroma_X_df
    # train labels from D index S1
    y_train = D['Cat'].iloc[S1_index]
    # test labels from D index everything
    y_test = D['Cat']
    
    make_U = chroma_X_df.drop(S1_index, axis=0)
    
    # used in find_candidate_set() & get_sorted_density_values()
    U = make_U.index.values
    # this is basically S1_index - used in find_candidate_set()
    L = X_train.index.values
    
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    
    print("SHAPES: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("Number of labelled instances, S1_index = ", len(S1_index))
    print("Number of unlabelled instances, U = ", len(U))
    
    # Base model used in initial investigations
    clf = SVC(kernel='rbf',
              C=50,
              gamma=1,
              class_weight='balanced')
    
    # clf = SVC(class_weight='balanced')

    # Fit model
    clf.fit(X_train, y_train) # .values.ravel()
    
    # BALANCED accuracy scores
    # y_pred_dec = clf.decision_function(X_test)
    y_true, y_pred = y_test, clf.predict(X_test)        
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print('Balanced accuracy on test set (y_true Vs y_pred: ', bal_acc)
    print("------------------------------------------------------------------")
    
# =============================================================================
#     # For AUC calculation
#     y_true_list.append(y_true)
#     y_pred_list.append(y_pred)
#     
#     # this is a python thing - I think if something is created in a global
#     # scope it gettings tracked globally, so anytime there's a reference to it
#     # every location gets updated. This was appending all slots with the
#     # latest iteration of S1_index - the syntax below means every slot has the
#     # corresponding S1_index for that Loop. I think the y_true and y_pred are
#     # different because they're local variables, or updated locally.
#     y_pred_dec_list.append(S1_index.copy())
# =============================================================================

    labelled_instances.append(X_train.shape[0])
    testAcc.append(bal_acc)

    # increment loop counter
    loop_counter = loop_counter + 1

    return U, L, loop_counter, y_true, y_pred, trainAcc, testAcc,\
        labelled_instances