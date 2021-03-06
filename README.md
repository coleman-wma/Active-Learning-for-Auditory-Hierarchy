# Experiment-3
Active Learning code for an experiment to assess methods for minimising the manual effort required to label audio instances with hierarchical labels.

#### agg_Clustering.py
Subjects the dataset to agglomerative clustering using the inverse of euclidean distance as a similarity metric. This process defines clusters of similar instances in across the feature space which is used in Exploration Guided Active Learning (EGAL) to select informative instances for labelling purposes. The code here also provides functions used to measure the distance between individual instances.

#### classification_accuracy.py
Experiment 3 also involved an analysis of a number of different feature representations. This code makes it easy to run a series of CV tests to compare performance.

#### egal_functions.py
A code implementation of the EGAL algorithm. EGAL identifies dense clusters of unlabelled instances which are furthest from already labelled instances. The selected instances can be presented to an oracle for labelling.

#### munging_functions.py
Various data munging functions to implement clustering, EGAL and USAL.

#### random_usc.py
This code implements a number of different selection methods which were ultimately compared with the EGAL selection method. Uncertainty Sampling Active Learning (USAL) uses model predictions to identify instances the model is most uncertain of as these are thought to be more informative for labelling purposes. Self Training approaches, where predicted labels were used in tandem with USAL were also implemented but found to be ineffective in this case. For baseline comparison a Random selection method was also implemented.

#### statistical_testing_II.py
Running statistical tests on the results obtained using the Wilcoxon rank sum test.

#### trainTest_functions.py
Model training functions.
