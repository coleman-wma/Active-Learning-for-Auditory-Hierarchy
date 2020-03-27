# Experiment-3
Active Learning code for an experiment to assess methods for minimising the manual effort required to label audio instances with hierarchical labels.

#### agg_Clustering.py
Subjects the dataset to agglomerative clustering using the inverse of euclidean distance as a similarity metric. This process defines clusters of similar instances in across the feature space which is used in Exploration Guided Active Learning (EGAL) to select informative instances for labelling purposes. The code here also provides functions used to measure the distance between individual instances.

#### classification_accuracy.py

#### egal_functions.py
A code implementation of the EGAL algorithm.

#### munging_functions.py
Various data munging functions to implement clustering, EGAL and USAL.

#### random_usc.py
This code implements an Uncertainty Selection Active Learning algorithm, which uses model predictions to identify instances the model is most uncertain of as these are thought to be more informative for labelling purposes than instances the model predicts confidently on.

#### statistical_testing_II.py
Running statistical tests on the results obtained using Wilcoxon signed-rank and Friedman tests.

#### trainTest_functions.py
Model training functions.
