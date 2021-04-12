#!/usr/bin/env python
# coding: utf-8
"""
Created on Sat Oct 24 14:52:17 2020

@author: kirksmi
"""
from sklearn.metrics import confusion_matrix
import xgboost
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             matthews_corrcoef, precision_score)
from sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.model_selection import StratifiedKFold
import warnings
import seaborn as sns
import copy
from sklearn.utils import class_weight
from sklearn.tree import DecisionTreeClassifier
from itertools import compress 
from sklearn.tree import plot_tree
from sklearn.tree._tree import TREE_LEAF
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss 
from matplotlib.backends.backend_pdf import PdfPages
import math
from sklearn.model_selection import GridSearchCV
import shap
from collections import Counter
from matplotlib.lines import Line2D
import matplotlib.colors as mcol
import matplotlib.cm as pltcm
shap.initjs()


def prune(tree):
    '''
    This function will get rid of repetitive branches in decision trees 
    which lead to the same class prediciton.
    Function written by GitHub user davidje13 (https://github.com/scikit-learn/scikit-learn/issues/10810)
    
    Function Inputs:
    ---------
    tree:   decision tree classifier
    
    Function Outputs:
    ---------
    tree:   pruned decision tree classifier
    '''
    tree = copy.deepcopy(tree)
    dat = tree.tree_
    nodes = range(0, dat.node_count)
    ls = dat.children_left
    rs = dat.children_right
    classes = [[list(e).index(max(e)) for e in v] for v in dat.value]
    leaves = [(ls[i] == rs[i]) for i in nodes]
    LEAF = -1
    for i in reversed(nodes):
        if leaves[i]:
            continue
        if leaves[ls[i]] and leaves[rs[i]] and classes[ls[i]] == classes[rs[i]]:
            ls[i] = rs[i] = LEAF
            leaves[i] = True
    return tree

def prune_index(inner_tree, index, threshold):
    '''
    This function will traverse a decision tree and remove any leaves with
    a count class less than the given threshold.
    Function written by David Dale
    (https://stackoverflow.com/questions/49428469/pruning-decision-trees)
    
    Function Inputs:
    ---------
    inner_tree:   tree object (.tree_) from decision tree classifier
    index:        where to start pruning tree from (0 for the root)
    threshold:    minimum class count in leaf 
    '''
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are children, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)
    
    
def make_confusion_matrix(y_true,
                          y_pred,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=(8,6),
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm
    using a Seaborn heatmap visualization.
    Basis of function from https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
   
    Function Inputs:
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUAR
    cf = confusion_matrix(y_true, y_pred)
    blanks = ['' for i in range(cf.size)]
    hfont = {'fontname':'Arial'}

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        
    #if it is a binary or multi-class confusion matrix
    if len(categories)==2:
        avg = "binary"
    else:
        avg = "macro"

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg)
    recall = recall_score(y_true, y_pred, average=avg)
    f1 = f1_score(y_true, y_pred, average=avg)
    mcc = matthews_corrcoef(y_true, y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    
    if sum_stats:
        stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nMCC={:0.3f}\nPearson's R={:0.3f}".format(
            accuracy,precision,recall,f1, mcc, r)
    else:
        stats_text = ""

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False
    
    if categories == 'auto':
        categories = range(len(categories))


    # MAKE THE HEATMAP VISUALIZATION
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font="Arial")
    ax = sns.heatmap(cf, annot=box_labels, fmt="",
                     cmap=cmap, cbar=cbar,
                     annot_kws={"size": 28})  #22
    ax.set_yticklabels(labels=categories, rotation=90, va="center",
                       fontsize=24, **hfont)
    ax.set_xticklabels(labels=categories,
                       fontsize=24, **hfont)   # 20

    # FORMATTING THE CONFUSION MATRIX LABELS/TEXT
    # if labels, put stats to right of CM
    if xyplotlabels:  
        plt.ylabel('True label', fontweight='bold', **hfont)
        plt.xlabel('Predicted label' + stats_text, fontweight='bold', **hfont)
    elif cbar:   # show color bar on right and stats below 
        plt.xlabel(stats_text, fontsize=15, **hfont)
    else:   # no color or labels, so put stats on right
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_label_coords(1.25,0.75)
        plt.ylabel(stats_text, fontsize=18, rotation=0, **hfont)#labelpad=75
    
    if title:
        plt.title(title, **hfont)
        
    plt.tight_layout()
    return ax


def xgb_func(X, y, num_iter, condition, class_names=None, depth="deep",
             imbalance="none"):
    '''
    This function trains an XGBoost model for predicting post-translational
    modifications (the target array) given a features matrix.
    
    For binary models, it is assumed that the 0 indicates Unregulated and 1
    represents the PTM class (e.g Acetyl or Phos)
    For multi-class models, it is assumed that the middle class is Unregulated
    and the lower/upper classes mark Phos and Acetyl, respectively
    (e.g -1=Phos, 0=Unreg, 1=Acetyl)
    
    The function uses RandomizedGridSearch within cross-validation to tune 
    the XGBoost hyperparameters and estimate model performance. The final model
    uses the hyperparameters from the best-scoring CV fold and is trained on 
    the entire dataset.
    
   
    Function Inputs:
    ---------
    1. X:            predictor features matrix 
    2. y:            target variable
    3. num_iter:     number of cross-validation folds
    4. condition:    string used to identify dataset condition (e.g "e.coli").
                     Used to name/save plots.
    5. class_names:  names of target classes   
    6. depth:        Determines the range of values for tuning the max_depth 
                     hyperparameter. Options are "deep" (default) or "shallow"
    7. imbalance     Determines the method for addressing class imbalance:
                     a.) List of two floats (for multi-class) or single float
                         (for binary): Model will use SMOTE oversampling for the
                         Phos/Acetyl classes. This assumes that Unregulated is
                         largest class. Ex: [0.5, 0.75] --> Phos class will be 
                         over-sampled to 50% of size of Unreg, Acetyl to 75% of
                         Unreg.
                     b.) "adasyn": uses Adasyn over-sampling method. Phos and                      
                         Acetyl classes over-sampled to 75% of Unreg class. 
                     c.) "undersample": undersamples larger classes to size of 
                         smallest class. 
                     d.) "none" (default): class balances are not adjusted
                     f.) "balanced": inverse proportion of classes are used to
                          assign class weights for "sample_weights" argument in 
                          hyperparameter tuning
           
    Function Outputs:
    ---------
    1. XGBoost model
    2. Dataframe w/ XGBoost cross-val scores
    '''
    
    # define font type for plots
    pltFont = {'fontname':'Arial'}
    
    # define feature names
    feat_names = X.columns
    
    # transform Target classes to 0, 1, 2
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = pd.Series(le.transform(y))
    
    # if class names not given, use class integers
    if class_names is None:
        class_names=[]
        for cl in y.unique():
            class_names.append(np.array2string(cl))
    num_class = len(np.unique(y))
    print("Number of class: {}".format(num_class))
   
    # hyperparameters to tune 
    # (max_depth adjusted based on 'depth' argument)
    if depth == "shallow":
        params={
        "learning_rate"    : [0.01, 0.05, 0.1, 0.3],
        "max_depth"        : range(4,6,1), #range(4,11,2),
        "min_child_weight" : [3, 5, 7],
        "subsample"        : [0.8, 0.9],
        "colsample_bytree" : [0.8, 0.9],
        }
    else:
        params={
        "learning_rate"    : [0.01, 0.05, 0.1, 0.3],
        "max_depth"        : range(4,11,2),
        "min_child_weight" : [3, 5, 7],
        "subsample"        : [0.8, 0.9],
        "colsample_bytree" : [0.8, 0.9],
        }

    ##### CV Analysis #####   

    # Define classifiers and hyperparameter search, based on binary vs multi-class problem
    if num_class == 2:  # binary model
        print("TRAINING BINARY MODEL!")
        # define classifier and hyperparameter tuning
        classifier = xgboost.XGBClassifier(objective='binary:logistic',
                                           n_estimators=150,
                                           use_label_encoder=False,
                                           eval_metric='logloss',
                                           random_state=123)
    
        random_search = RandomizedSearchCV(classifier, param_distributions=params,
                                          n_iter=30, scoring='f1',  # 100
                                          n_jobs=-1, cv=5, verbose=3,
                                          random_state=123) 
        avg = "binary"
    
    elif num_class > 2: # multi-class model
        print("TRAINING MULTI-CLASS MODEL!")
        classifier = xgboost.XGBClassifier(objective='multi:softmax',
                                           n_estimators=150,
                                           use_label_encoder=False,
                                           num_class=num_class,
                                           eval_metric='mlogloss',
                                           random_state=123) #multi:softmax

        random_search = RandomizedSearchCV(classifier, param_distributions=params,
                                          n_iter=30, scoring='f1_macro',  # 100
                                          n_jobs=-1, cv=5, verbose=3,
                                          random_state=123) 
        avg = "macro"
        
    # Stratified cross-val split
    cv = StratifiedKFold(n_splits=num_iter,
                         shuffle=True, 
                         random_state=123)
    
    # create empty lists to store CV scores, confusion mat, etc.
    acc_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    mcc_list = []
    r_list = []
    
    y_test = []
    y_pred = []
    cmCV = np.zeros((num_class, num_class))
    
    paramDict = {}

    count = 0
    
    # loop through cross-val folds
    for train_index, test_index in cv.split(X, y):
        X_trainCV, X_testCV = X.iloc[train_index], X.iloc[test_index]
        y_trainCV, y_testCV = y.iloc[train_index], y.iloc[test_index]
            
        # train and fit model according to the desired class imbalance method
        if isinstance(imbalance, (list, float)):
            class_values = y_trainCV.value_counts()
            if num_class > 2:
                smote_dict = {0:int(round(class_values[1]*imbalance[0])),
                              1:class_values[1],
                              2:int(round(class_values[1]*imbalance[1]))}
            else:
                smote_dict = {0:class_values[0],
                              1:int(round(class_values[0]*imbalance))}
                
            print(smote_dict)
            oversample = SMOTE(sampling_strategy=smote_dict)
            X_trainCV, y_trainCV = oversample.fit_resample(X_trainCV, y_trainCV)
            random_search.fit(X_trainCV, y_trainCV)     
            
        elif imbalance=="adasyn":
            class_values = y_trainCV.value_counts()
            smote_dict = {0:int(round(class_values[1]*0.75)),
                          1:class_values[1],
                          2:int(round(class_values[1]*0.75))}
            ada = ADASYN(sampling_strategy = smote_dict,
            random_state=123, n_neighbors=10)
            X_trainCV, y_trainCV = ada.fit_resample(X_trainCV,y_trainCV)
            random_search.fit(X_trainCV, y_trainCV)

        elif imbalance=="undersample":
            nr = NearMiss() 
            X_trainCV, y_trainCV = nr.fit_sample(X_trainCV, y_trainCV)
            random_search.fit(X_trainCV, y_trainCV)
        
        elif imbalance=="none":
            random_search.fit(X_trainCV, y_trainCV)
            
        elif imbalance=="balanced":            
            weights = class_weight.compute_sample_weight("balanced", y_trainCV)
            
            random_search.fit(X_trainCV, y_trainCV,
                              sample_weight=weights)
        # get best estimator from random search
        randomSearch_mdl = random_search.best_estimator_
            
        # tune gamma and get new best estimator
        params_gamma = {'gamma':[0, 0.1, 0.3, 0.5]}
        gamma_search = GridSearchCV(estimator = randomSearch_mdl, 
                            param_grid = params_gamma, scoring='f1_macro',
                            n_jobs=-1 , cv=3)
        gamma_search.fit(X_trainCV, y_trainCV)
        best_Mdl = gamma_search.best_estimator_
        
        # print and store best params for current fold
        print("Cross-val Fold {}, Model Params: {}".format(count, best_Mdl))
        paramDict[count] = best_Mdl.get_params
        
        # make model predictions on X_testCV and store results
        y_predCV = best_Mdl.predict(X_testCV)        
        y_test.extend(y_testCV)
        y_pred.extend(y_predCV)
        
        cm = confusion_matrix(y_testCV, y_predCV)
        print("current cm: \n",cm)
        # update overal confusion mat
        cmCV = cmCV+cm
        print("Combined cm: \n", cmCV)
        
        # calculate classification scores and store
        accuracy = accuracy_score(y_testCV, y_predCV)
        f1 = f1_score(y_testCV, y_predCV, average=avg)
        recall = recall_score(y_testCV, y_predCV, average=avg)
        precision = precision_score(y_testCV, y_predCV, average=avg)
        mcc = matthews_corrcoef(y_testCV, y_predCV)
        r = np.corrcoef(y_testCV, y_predCV)[0, 1]
            
        acc_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        mcc_list.append(mcc)
        r_list.append(r)
    
        count = count+1
             
    # print final confusion mat
    print("final CV confusion matrix: \n",cmCV)
   
    ### plot confusion matrix results 
    path = '../figures/crossval/'
    try:
        os.makedirs(path)
    except OSError:
        print("Directory already created")
        
    make_confusion_matrix(y_test, y_pred, figsize=(8,6), categories=class_names,
                          xyplotlabels=True, cbar=False, sum_stats=False)
    plt.ylabel("Experimental Labels", fontsize=24)
    plt.xlabel("Predicted Labels", fontsize=24)
    plt.tight_layout()
    plt.savefig("../figures/crossval/{}_XGBcrossval_confusionMat.png".format(condition),
                   dpi=600)
    plt.show()  
    
    # get average scores
    Accuracy = np.mean(acc_list)
    F1 = np.mean(f1_list)
    Precision = np.mean(precision_list)
    Recall = np.mean(recall_list)
    MCC = np.mean(mcc_list)
    Corr = np.mean(r_list)
    
    scores = [Accuracy, Recall, Precision, F1, MCC, Corr] #AUC,
    
    # get stats for CV scores
    loop_scores = {'Accuracy':acc_list,
                   'Recall':recall_list,
                   'Precision':precision_list,
                   'F1':f1_list,
                   'MCC':mcc_list,
                   'R':r_list}
                    
    df_loop_scores = pd.DataFrame(loop_scores)
    print("Model score statistics: ")
    loop_stats = df_loop_scores.describe()
    print(loop_stats)
    
    # plot CV scores
    plt.rcParams.update(plt.rcParamsDefault) 
    plt.rcParams['xtick.major.pad']='10'
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(df_loop_scores.columns, scores,
           yerr=loop_stats.loc['std',:],
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10,
           width=0.8)
    ax.set_ylim([0, 1.0])
    plt.yticks(**pltFont)
    ax.set_xticks(df_loop_scores.columns)
    ax.set_xticklabels(df_loop_scores.columns,**pltFont,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig('../figures/crossval/{}_XGB_crossVal_barGraph.png'.format(condition),
                bbox_inches='tight', dpi=600)
    plt.show()

    # create dataframe with mean scores
    data = {'Metric':['Acc', 'Recall', 'Precision','F1', 'MCC', 'PearsonsR'], 
      'Scores':[Accuracy, Recall, Precision, F1, MCC, Corr]} 
    df_scores = pd.DataFrame(data)
    df_scores = df_scores.set_index('Metric')

    ### train model on entire training dataset using params from best CV model     
    maxpos = mcc_list.index(max(mcc_list))
    final_params = paramDict[maxpos]
    print("CV MCCs: {}".format(mcc_list))
    print("Best parameters: ", final_params)
    final_Mdl = classifier
    final_Mdl.get_params = final_params
    
    if isinstance(imbalance, (list, float)):
        class_values = y.value_counts()
        if num_class > 2:
            smote_dict = {0:int(round(class_values[1]*imbalance[0])),
                          1:class_values[1],
                          2:int(round(class_values[1]*imbalance[1]))}
        else:
            smote_dict = {0:class_values[0],
                          1:int(round(class_values[0]*imbalance))}

        print(smote_dict)
        oversample = SMOTE(sampling_strategy=smote_dict)
        X, y = oversample.fit_resample(X, y)
        final_Mdl.fit(X, y)
        
    elif imbalance=="adasyn":
        class_values = y.value_counts()
        smote_dict = {0:int(round(class_values[1]*0.75)),
                          1:class_values[1],
                          2:int(round(class_values[1]*0.75))}
        ada = ADASYN(sampling_strategy = smote_dict,
                     random_state=123, n_neighbors=10)
        X, y = ada.fit_resample(X, y)
        final_Mdl.fit(X, y)
            
    elif imbalance=="undersample":
        X, y = nr.fit_sample(X, y)
        final_Mdl.fit(X, y)
        
    elif imbalance=="none":
        final_Mdl.fit(X, y)
        
    elif imbalance=="balanced":
        w = class_weight.compute_sample_weight("balanced", y)
        final_Mdl.fit(X, y, sample_weight=w)


    ### Feature importances
    importances = final_Mdl.feature_importances_
     # Sort in descending order
    indices = np.argsort(importances)[::-1]
     # Rearrange feature names so they match the sorted feature importances
    names = [feat_names[i] for i in indices]   # for sfs
    
     # Create plot
    plt.figure()
    plt.bar(range(X.shape[1]), importances[indices]) 
    plt.title("XGBoost Feature Importance")
    plt.xticks(range(X.shape[1]), names,
                fontsize=18, rotation=45, horizontalalignment="right")
    plt.yticks(fontsize=20)
    plt.bar(range(X.shape[1]), importances[indices])  
    plt.savefig("../figures/crossval/{}_XGB_featureImps.png".format(condition),
                  bbox_inches='tight', dpi=600)
    plt.show()
    
    return final_Mdl, loop_stats


def multi_heatmap(explainer, X, num_class, order="explanation",
                  feat_values=None, cmap="bwr", class_names=None,
                  max_display=10, condition=None, save=True,
                  showOutput=True):
    '''
    This function creates a heatmap of SHAP values, given a SHAP explainer object
    and the feature matrix to be explained. The function is primarily made up
    of the source code from the shap.plots.heatmap function, however it has been
    adjusted to accomodate multi-class models. 
    Refer to https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/heatmap.html
    for more info on the SHAP heatmaps.
    
    Function Inputs:
    ----------
    1. explainer:   SHAP explainer object
    2. X:           The independent data that you want explained 
    3. num_class:   Number of classes in model
    4. class_names: Names of target classes 
    5. order:       Method for clustering the SHAP values
                    a.) "explanation" (default): samples are grouped based on 
                         a hierarchical clustering by their explanation similarity
                    b.) "output": samples are ordered by model output, f(x), 
                        which is shown in log odds on the line plot above the 
                        heatmap.
    6. feat_values: Array of values used to sort the model features on the y-axis.
                    Should be equal to number of features. If not given, features
                    are sorted by their absolute magnitude SHAP value.
    7. cmap:        Colormap used for the heatmap. Default is 'bwr'.
    8. max_display: Maximum number of features to display on y-axis.
    9. condition:   String used to identify dataset condition (e.g "e.coli").
                    Used to name/save plots.
    10. save:       Whether to save the heatmap figures. Default is True.
    11. showOutput  Whether to show model output, f(x), in lineplot above 
                    heatmap.
    
            
    Function Outputs: 
    ----------
    1. explainer:   SHAP explainer object
    2. shap_values: matrix of SHAP values
    '''
    pltFont = {"fontname":"Arial"}
    f = []
    for class_num in range(num_class):
        feature_values=feat_values
        
        print("ITERATION {}".format(class_num))
        shap_values = explainer(X)[:,:,class_num]
        
        # define clustering method for observations
        if order == "explanation":
            instance_order = shap_values.hclust()
        elif order=="output":   # use "output" to cluster by model output
            instance_order = shap_values.sum(1)
            instance_order = instance_order.argsort.flip.values

        # define order of features on y-axis
        if feat_values is None:
            feature_values = shap_values.abs.mean(0)
            feature_values = feature_values.values
        show=True

        # sort the SHAP values matrix by rows and columns
        values = shap_values.values
        
        feature_order = np.argsort(-feature_values)
        
        xlabel = "Instances"

        feature_names = np.array(shap_values.feature_names)[feature_order]
        values = shap_values.values[instance_order][:,feature_order]
        feature_values = feature_values[feature_order]
        
        # collapse
        if values.shape[1] > max_display:
            new_values = np.zeros((values.shape[0], max_display))
            new_values[:, :max_display-1] = values[:, :max_display-1]
            new_values[:, max_display-1] = values[:, max_display-1:].sum(1)
            new_feature_values = np.zeros(max_display)
            new_feature_values[:max_display-1] = feature_values[:max_display-1]
            new_feature_values[max_display-1] = feature_values[max_display-1:].sum()
            feature_names = list(feature_names[:max_display])
            feature_names[-1] = "Sum of %d other features" % (values.shape[1] - max_display + 1)
            values = new_values
            feature_values = new_feature_values
        
        # define the plot size
        # cmap='coolwarm'
        plt.figure()
        row_height = 0.5
        plt.gcf().set_size_inches(8, values.shape[1] * row_height + 2.5)
        
        # plot the matrix of SHAP values as a heat map
        vmin = np.nanpercentile(values.flatten(), 1)
        vmax = np.nanpercentile(values.flatten(), 99)
        plt.imshow(
            values.T, aspect=0.7 * values.shape[0]/values.shape[1], interpolation="nearest", vmin=min(vmin,-vmax), vmax=max(-vmin,vmax),
            cmap=cmap)
        yticks_pos = np.arange(values.shape[1])
        yticks_labels = feature_names
        
        # plot f(x) above heatmap
        if showOutput:
            plt.yticks([-1.5] + list(yticks_pos),
                       ["f(x)"] + list(yticks_labels),
                       fontsize=18, **pltFont)          # for y-axis labels
            plt.ylim(values.shape[1]-0.5, -3)
            plt.xticks(fontsize=18, **pltFont)     # for x-axis ticks
            # create model output plot above heatmap
            plt.gca().xaxis.set_ticks_position('bottom')
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.axhline(-1.5, color="#aaaaaa", linestyle="--", linewidth=0.5)
            fx = values.T.mean(0)
            f.append(fx)
            plt.plot(-fx/np.abs(fx).max() - 1.5, color="#000000", linewidth=1)
            
        else:
            plt.yticks(list(yticks_pos),
                       list(yticks_labels), fontsize=15, **pltFont)
            
            plt.ylim(values.shape[1]-0.5, -3)
            plt.xticks(fontsize=14, **pltFont)
            fx = values.T.mean(0)
            f.append(fx)
            
        
        #pl.colorbar()
        plt.gca().spines['left'].set_bounds(values.shape[1]-0.5, -0.5)
        plt.gca().spines['right'].set_bounds(values.shape[1]-0.5, -0.5)
        # plot feature importance bars to right of heatmap
        b = plt.barh(
            yticks_pos, (feature_values / np.abs(feature_values).max()) * values.shape[0] / 20, 
            0.7, align='center', color="#000000", left=values.shape[0] * 1.0 - 0.5
            #color=[colors.red_rgb if shap_values[feature_inds[i]] > 0 else colors.blue_rgb for i in range(len(y_pos))]
        )
        for v in b:
            v.set_clip_on(False)
        plt.xlim(-0.5, values.shape[0]-0.5)
        plt.xlabel(xlabel, fontsize=20, **pltFont) # for "Instances"
        
        
        if True:   # plot colorbar
            import matplotlib.cm as cm
            m = cm.ScalarMappable(cmap=cmap)
            m.set_array([min(vmin,-vmax), max(-vmin,vmax)])
            cb = plt.colorbar(m, ticks=[min(vmin,-vmax), max(-vmin,vmax)], aspect=1000, fraction=0.0090, pad=0.10,
                            panchor=(0,0.05))
            #cb.set_ticklabels([min(vmin,-vmax), max(-vmin,vmax)])
            cb.set_label("SHAP value", size=20, labelpad=-30, **pltFont)   # orig pad: -10, for colorbar label
            cb.ax.tick_params(labelsize=18, length=0)   # orig size: 11, for colorbar numbers
            cb.set_alpha(1)
            cb.outline.set_visible(False)
            bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
            cb.ax.set_aspect((bbox.height - 0.9) * 15)
            cb.ax.set_anchor((1,0.2))
            #cb.draw_all()
            
        for i in [0]:
            plt.gca().get_yticklines()[i].set_visible(False)
            
        if class_names:
            plt.title("{}".format(class_names[class_num]),fontsize=20,pad=0)
        
        # save 
        if save:
            if class_names is None:
                plt.savefig("../figures/SHAP/heatmaps/ShapHeatmap{}.png".format(i),
                              dpi=600, bbox_inches='tight')
            else:
                plt_name = "ShapHeatmap_{}".format(class_names[class_num])
                print(plt_name)
                plt.savefig("../figures/SHAP/heatmaps/{}_{}.png".format(plt_name, condition),
                            dpi=600, bbox_inches='tight')
        if show:
            plt.show()
            
    return 

    

 ##### SHAP ANALYSIS #####
 
def shap_func(xgbModel, X, condition, class_names):
    '''
    This function performs various analyses using the SHAP package. Given
    a classification model and a dataframe of feature values which you want
    explained, the function will output the SHAP explainer object and SHAP
    values. Additionally, SHAP summary plots, heatmaps and decision plots are
    created. 
    Refer to https://shap.readthedocs.io/en/latest/ for more info on the SHAP
    package.
    
    Function Inputs:
    ----------
    1. xgbModel:    XGBoost classification model
    2. X:           the independent data that you want explained 
    3. condition:   string used to identify dataset condition
    4. class_names: names of target classes 
            
    Function Outputs: 
    ----------
    1. explainer:   SHAP explainer object
    2. shap_values: matrix of SHAP values
    '''
    
    # create directories for SHAP figures
    paths = ["../figures/SHAP",
             "../figures/SHAP/summary_plots",
             "../figures/SHAP/dependence_plots",
             "../figures/SHAP/heatmaps"]
    for path in paths: 
        try:
            os.mkdir(path)
        except OSError:
            print ("Directory %s failed (may already exist)" % path)
        else:
            print ("Successfully created the directory %s " % path)
    
    # define font type for SHAP plots
    pltFont = {'fontname':'Arial'}
    plt.rcParams.update(plt.rcParamsDefault)    

    # define feature and class names
    feat_names = X.columns
    if class_names is None:
        class_names = xgbModel.classes_
    num_class = len(class_names)

    
    # create SHAP explainer and get SHAP values
    explainer = shap.TreeExplainer(xgbModel,
               feature_perturbation='tree_path_dependent') 
    X_test = X
    shap_values = explainer.shap_values(X_test)  
    expected_value = explainer.expected_value

    ## SHAP summary and dependence plots
    
    # combined
    if num_class > 2:
        # combined summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, class_names=class_names, show=False) #class_names = multi_classes
        plt.title("SHAP Summary Plot: {}".format(condition), fontsize=20, **pltFont)
        plt.yticks(fontsize=18, **pltFont)
        plt.xticks(fontsize=18, **pltFont)
        plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)",
                   fontsize=20, **pltFont)
        plt.savefig("../figures/SHAP/summary_plots/{}_MultiSummaryPlot.png".format(condition),
             bbox_inches='tight', dpi=600)
        plt.show()
        
        # individual class summary plots
        for which_class in range(num_class):
            print("Current class: ", which_class)
  
            # summary single class
            shap.summary_plot(shap_values[which_class], X_test,
                              color_bar=False, show=False)
            plt.title("{}".format(class_names[which_class]),fontsize=20, **pltFont)
            plt.yticks(fontsize=18, **pltFont)
            plt.xticks(fontsize=18, **pltFont)
            plt.xlabel("SHAP Value (impact on model output)",fontsize=20, **pltFont)
            # make our own color bar so that we can adjust font/label sizes
            cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["#4d73ff","#ff1303"])
            cnorm = mcol.Normalize(vmin=0,vmax=1)
            m = pltcm.ScalarMappable(norm=cnorm,cmap=cm1)
            m.set_array([0, 1])
            cb = plt.colorbar(m, ticks=[0, 1], aspect=1000)
            cb.set_ticklabels(["Low", "High"])
            cb.set_label("Feature Value", size=18, labelpad=-10)
            cb.ax.tick_params(labelsize=18, length=0)
            cb.set_alpha(1)
            cb.outline.set_visible(False)
            bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
            cb.ax.set_aspect((bbox.height - 0.9) * 20)
            plt.savefig("../figures/SHAP/summary_plots/{}_{}_SingleSummaryPlot.png".format(condition, class_names[which_class]),
                          dpi=600, bbox_inches='tight')
            plt.show()
             
            ## dependence plots (only plot for top 3 important feats)
            vals = np.abs(shap_values[which_class]).mean(0)
            # Sort feature importances in descending order
            indices = np.argsort(vals)[::-1]
            feat_names = X_test.columns
            sorted_names = [feat_names[ind] for ind in indices]  
            
            for i in range(3):   
                shap.dependence_plot(ind=sorted_names[i],
                                 shap_values=shap_values[which_class], 
                                 features=X_test,
                                 interaction_index="auto",
                                 x_jitter=1,
                                 show=False) # *****
                # plt.title("Dependence Plot")
                plt.savefig("../figures/SHAP/dependence_plots/{}_{}_{}_dependencePlot.png".format(
                    condition, class_names[which_class], sorted_names[i]))
                plt.show()

                 
    elif num_class == 2:
         # summary plot for 'positive' class
         plt.figure()
         shap.summary_plot(shap_values, X_test, show=False) 
         plt.title("SHAP Summary Plot: {}".format(condition),fontsize=15)
         plt.show()
         plt.savefig("../figures/SHAP/{}_SummaryPlot.png".format(condition),
                      bbox_inches='tight', dpi=600)
   
         shap.dependence_plot("rank(0)", shap_values, X_test, show=False) 
         plt.title("Dependence Plot",fontsize=15)
         plt.savefig("../figures/SHAP/dependence_plots/{}_DependencePlot.png".format(condition),
                     dpi=600)
         plt.show()
      
    
    ## SHAP heatmap 
    if num_class == 2:
        explainer_shaps = explainer(X_test)    # X_test  
        shap.plots.heatmap(explainer_shaps,
                           max_display=len(feat_names))
    elif num_class > 2:
        multi_heatmap(explainer, X_test, num_class, order="output",
                      class_names=class_names,
                      condition=condition,
                      max_display=len(feat_names))
          
    return explainer, shap_values



def predict_genes(X, y, all_genes, select_genes, xgb_clf,
                  explainer, class_names, condition="PredictGenes"):
    '''
    This function is used to analyze a model's predictions on specific genes
    of interest. It is assumed that the true class is known for the dataset.
    
    Function Inputs:
    ----------
    1. X:            predictor features
    2. y:            target variable
    3. all_genes:    string array of all genes corresponding to the rows
                     in the X and y datasets              
    4. select_ganes: string or string array of gene names to be analyzed
    5. xgb_clf:      Classifier model
    6. explainer:    SHAP explainer object
    7. class_names:  string array of class names
    8. condition:    string of dataset condition (for naming plots)   
            
    Function Outputs: 
    ----------
    1. SHAP decision plots - shows the prediction path for the all instances of the
       gene of interest, with a separate plot for each class.
    2. SHAP mulitoutput decision plot - shows the prediction path for a single 
       instance of the gene of interest, but all classes are displayed on same
       plot.
       
      All plots are saved to the 'figures/predict_genes' folder.
    '''
    # make folder for figures
    path= "../figures/predictGenes"
    try:
        os.mkdir(path)
    except OSError:
        print ("Directory %s failed (may already exist)" % path)
    else:
        print ("Successfully created the directory %s " % path)
    pltFont = {'fontname':'Arial'}

    # get feature names and number of classes
    feature_names = X.columns
    feat_names_list = feature_names.tolist()
    num_class = len(y.unique())
    
    # transform class integers to [0,1,2]
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = pd.Series(le.transform(y))
    
    # get indices of gene of interest
    i_genes = []
    for gene in select_genes:
        bool_list = all_genes["genes"].eq(gene)
        i_gene = list(compress(range(len(bool_list)), bool_list))
        i_genes.extend(i_gene)
        
    # get X, y and y_pred for genes of interest
    X_test = X.loc[X.index.isin(i_genes), feature_names]
    y_test = y[y.index.isin(i_genes)]
    y_pred = xgb_clf.predict(X_test) 
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # create dataframe showing y_test and y_pred
    test_index = X_test.index.tolist()
    test_genes = all_genes.loc[test_index,"genes"]
    test_rxns = all_genes.loc[test_index,"reaction"]
    data = {'Test Genes': test_genes,
            'Test Rxns': test_rxns,
            'y_test': y_test,
            'y_pred': y_pred}
    df = pd.DataFrame (data, columns = ['Test Genes','Test Rxns','y_test','y_pred'])
    print(df)

    ### SHAP decision plot 
    expected_value = explainer.expected_value
    print(expected_value)
    
    for gene in select_genes:
        
        bool_genes = df["Test Genes"]== gene   # get rows for current gene
        
        features=X_test[bool_genes]   # get features for current gene
        
        # y_pred (from XGBoost mdl) and y_test for current gene
        y_pred_select = y_pred[bool_genes]
        y_test_select = y_test[bool_genes]
        
        # probabilities for current gene prediction
        y_proba = xgb_clf.predict_proba(features)
        print("Probabilities:")
        print(y_proba)
        
        # log odds for current gene prediction
        logodds = xgb_clf.predict(features, output_margin=True)
        print("Log odds:")
        print(logodds)
        
        misclass_genes = y_test_select != y_pred_select    # mis-classified genes (to be used for highlight, if desired)
        
        # we will create decision plot for each class, so loop through classes
        for j in y.unique()[::-1]:
            # ID genes of current class in y_test (to be used for highlighting)
            class_genes = y_test_select == j   
        
            # get shap values of specified observations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values_decision = explainer.shap_values(features)[j]   # j starts with -1, so add 1 to get index 0
                shap_interaction_values = explainer.shap_interaction_values(features)[j]
            if isinstance(shap_interaction_values, list):
                shap_interaction_values = shap_interaction_values
      
            # create decision plot
            shap.decision_plot(expected_value[j], shap_values_decision,
                               features,
                               highlight=class_genes,
                               show=False)
            plt.title("SHAP Decision Plot: {}, {}".format(gene, class_names[j]), fontsize=15)
            plt.savefig(path+"{}_{}_{}DecisionPlot.png".format(condition, gene, class_names[j]),
                        bbox_inches='tight', dpi=600)
            plt.show()
            
            
    ### SHAP multi-output decision plot
    
    # function create legend labels using class name and log odds value
    def class_labels(row_index):
            return [f'{j} ({logodds[row_index, i].round(2):.2f})' for i,j in enumerate(class_names)]
    
    for gene in select_genes:   
        bool_genes = (df["Test Genes"] == gene) & (df["y_test"]==df["y_pred"])   # ID rows with current gene and correct classification
        
        # if at leat 1 correct prediction, proceed
        if sum(bool_genes) > 0:
            if sum(bool_genes) == 1:
                rows = 1   # explain 1st two observations for each gene
            else:
                rows = 2
        
            features=X_test[bool_genes]   # get features for above rows
            rxns = df.loc[bool_genes,"Test Rxns"]
            print("REACTIONS: ", rxns)
            shap_values = explainer.shap_values(features)   # get shap values for select features
            shap_explainer = explainer(features)
    
            # get log odds for select genes
            logodds = xgb_clf.predict(features, output_margin=True)
            print("{} Log odds: ".format(gene))
            print(logodds)
            
            # create multi-output decision plot for first 2 observations (w/ correct classification)
            for row in range(rows):
                rxn = rxns.iloc[row]
                plt.figure(figsize=(8,6))
                shap.multioutput_decision_plot(expected_value, shap_values,
                                               row_index=row,
                                               feature_names=feat_names_list,
                                               highlight=[np.argmax(logodds[row])],
                                               legend_labels=None,
                                               show=False)
                # fix line colors and weights
                num_lines=len(plt.gca().lines)
                num_classes=len(class_names)
                for line, color in zip(plt.gca().lines[num_lines-num_classes:num_lines],
                                       ["#0070C0","#FFD55A","#6DD47E"]):
                    line.set_linewidth(4)
                    line.set_color(color)
                # fix legend
                plt.legend(handles=plt.gca().lines[num_lines-num_classes:num_lines],
                           labels=class_labels(row),
                           loc="lower right", fontsize=16)
                plt.title("Gene-Rxn: {}-{}".format(gene, rxn), fontsize=20, **pltFont)
                plt.xticks(fontsize=20, **pltFont)
                plt.yticks(fontsize=20, **pltFont)
                plt.xlabel("Model output value", fontsize=20, **pltFont)
                
                regType = class_names[np.argmax(logodds[row])]
                plt.savefig(path+"{}_{}{}_{}_MultiOutputPlot.png".format(
                    condition, gene, row, regType),
                    bbox_inches='tight', dpi=600)        
                plt.show()
                

def decisionTree(X, y, class_names, weighting="balanced", weights=None, pruneLevel=0, condition="Condition", save=True):
    '''
    Function Inputs:
    ----------
    1. X:            dataframe of features matrix
    2. y:            target variable array
    3. class_names:  string array of class names
    4. weighting:    options for handling class imbalance
                     a.) none: no class weights or sampling applied
                     b.) balanced (default): inverse class proportions
                         used to assign class weights
                     c.) smote: SMOTE over-sampling applied 
                     d.) tuned: weights are tuned via cross-validation
    5. weights       dictionary of weights to use with the "tuned" weighting
                     option   
    6. pruneLevel    integer, designating the minimum number of observations
                     in a leaf for it to be pruned  
    7. condition:    string of dataset condition (for naming plots/files)
                     
            
    Function Outputs: 
    ----------
    Decision tree plots are saved to the 'figures/decision_trees' folder.
'''
    # define path name for saving trees
    path = "../figures/decision_trees"
    
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
        
    # create Dtrees at multiple depths
    treeDict_VarDepths = {}
    
    param_dist = {"min_samples_leaf": [5,10,20,30,50],# default=1, range:1-20
                  "min_samples_split": [5,10,20,30,50], # default=2, range:1-40
                  "criterion": ["gini", "entropy"],
                  "max_features": [0.4, 0.6, 0.8, 1.0]}
    
    # define weights if not given 
    if weights is None:
        weights = [{-1:2,0:1,1:2},{-1:4,0:1,1:4},
                   {-1:4,0:1,1:2},{-1:2,0:1,1:4}]
        
    feats = X.columns
    
    for i, depth in enumerate([3, 4]):
    
        if weighting == "tuned":
            treeDict_VarWeights = {}
            scores = []
            for count, weight in enumerate(weights):
              
                # Instantiate a Decision Tree classifier: tree
                dtree = DecisionTreeClassifier(class_weight=weight,
                                               max_depth=depth,
                                               random_state=123)
                
                # Instantiate the RandomizedSearchCV object: tree_cv
                tree_cv = RandomizedSearchCV(dtree, param_dist,
                                             n_iter=30, cv=5,
                                             scoring="f1_macro",
                                             random_state=123)  
                    
                # Fit it to the data
                tree_cv.fit(X,y) #,sample_weight=w_array
                treeDict_VarWeights[count] = tree_cv
                scores.append(tree_cv.best_score_)
                print("Best score is {}".format(tree_cv.best_score_))
        
            maxpos = scores.index(max(scores)) 
            print(treeDict_VarWeights[maxpos].best_params_)
            tree_clf = treeDict_VarWeights[maxpos].best_estimator_
            
        elif weighting == "balanced":
            # Instantiate a Decision Tree classifier: tree
            dtree = DecisionTreeClassifier(class_weight='balanced',
                                           max_depth=depth,
                                           random_state=123)
            # Instantiate the RandomizedSearchCV object: tree_cv
            tree_cv = RandomizedSearchCV(dtree, param_dist,
                                         n_iter=50, cv=3,
                                         scoring="f1_macro",
                                         random_state=123)  
            # Fit it to the data
            tree_cv.fit(X ,y,
                        sample_weight=class_weight.compute_sample_weight("balanced", y))
            # get best estimator from random search
            tree_clf=tree_cv.best_estimator_
            
        elif weighting == "smote":
            # Instantiate a Decision Tree classifier: tree
            dtree = DecisionTreeClassifier(max_depth=depth,
                                           random_state=123)
            
            # Fit it to the data
            oversample = SMOTE()
            XtrainRes, ytrainRes = oversample.fit_resample(X, y)
            # Instantiate the RandomizedSearchCV object: tree_cv
            tree_cv = RandomizedSearchCV(dtree, param_dist,
                                         n_iter=50, cv=5,
                                         scoring="f1_macro",
                                         random_state=123)  
            tree_cv.fit(XtrainRes, ytrainRes)
            # get best estimator from random search
            tree_clf=tree_cv.best_estimator_
        
        elif weighting == "none":
            # Instantiate a Decision Tree classifier: tree
            dtree = DecisionTreeClassifier(max_depth=depth,
                                           random_state=123)
            
            # Instantiate the RandomizedSearchCV object: tree_cv
            tree_cv = RandomizedSearchCV(dtree, param_dist,
                                         n_iter=50, cv=3,
                                         scoring="f1_macro",
                                         random_state=123)  
            # Fit it to the data
            tree_cv.fit(X,y)
            # get best estimator from random search
            tree_clf=tree_cv.best_estimator_
        
        # get rid of redundant splits
        tree_clf = prune(tree_clf)
        
        # evaluate model w/ resubstitution
        y_predDT = tree_clf.predict(X)
        mcc = matthews_corrcoef(y, y_predDT)
        print(mcc)
    
        ### prune tree and re-evaluate
        prunedTree_clf = copy.deepcopy(tree_clf)  # create copy of DT to prune
        prune_index(prunedTree_clf.tree_, 0, pruneLevel)   # run prune function
        prunedTree_clf = prune(prunedTree_clf)  # get rid of unnecessary splits
        
        # store Dtree model 
        treeDict_VarDepths[i] = prunedTree_clf
    
        # evaluate model w/ resubstitution
        yPredPruned = prunedTree_clf.predict(X)
        mccPruned = matthews_corrcoef(y, yPredPruned)
        print("Pruned tree MCC: {}".format(mccPruned))
        
        cm = confusion_matrix(y, yPredPruned)
        print(cm)

        ### save tree figure                
        plt.rcParams.update(plt.rcParamsDefault)    

        suffix = '.pdf'
        file_name = "/{}_MaxDepth{}".format(
            condition, depth)
        
        pp = PdfPages(path+file_name+suffix)
        
        fig=plt.figure(figsize=(40,20))
        
        plt.subplot2grid((4, 4), (0, 3), colspan=1,rowspan=1)
        sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                            annot=True, cmap='Blues', fmt='d')
        
        plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
        plot_tree(prunedTree_clf, 
                          feature_names=feats, 
                          class_names=class_names, 
                          filled=True, 
                          rounded=True,
                          fontsize=11,
                          impurity=False) 
                          
        if weighting == "tuned":
            plt.title('Pruned: {} \n  Max. Depth: {} \n Weights: {} \n Resubstitution MCC: {:.3f}'.format(
               condition, depth, weights[maxpos], mccPruned))#%%
        else:
            plt.title('Pruned: {} \n  Max. Depth: {} \n Weights: Balanced \n Resubstitution MCC: {:.3f}'.format(
               condition, depth, mccPruned),loc="left")
        plt.show()
        
        if save:
            pp.savefig(fig, bbox_inches='tight', dpi=600)
        
        pp.close()
    
        
def mdl_predict(mdl, X, condition, gene_reactions,
                class_names=None, pos_label=0,
                y=None, confusion_mat=True,
                bar=False, swarm=False,
                pairwise=False, boxplot=False,
                explainer=None):
    '''
    Function Inputs:
    ----------
    1. mdl:             Classifier model
    2. X:               Features matrix used for predictions
    3. y:               True values of target variable. If given, output will
                        include figures related to classification performance.
                        If NOT given, model will only return predictions.             
    4. condition:       string of dataset condition (for naming plots/files)
    5. class_names:     Target variable class names
    6. gene_reactions:  Dataframe with gene and reaction IDs for the features
                        matrix. Used to output names of predictions for each class.
    7. confusion_mat:   Option to output classification confusion matrix
                        (default=True).
    8. bar:             Option to output bar graph with classification scores
                        (default=False).
    9. swarm:           Option to output matrix of swarmplots for all numerical
                        features, grouped by true pos and false neg (default=False).
    10. pairwise:       Option to output pairwise plot for top 5 most important
                        features (default=False).
    11. boxplot:        Option to output matrix of feature boxplots grouped by 
                        features classification group (default=False).
    12. explainer:      SHAP explainer object (default=None). If given, several
                        plots are produced with the SHAP package.  
    Function Outputs:
    ----------
    *** If 'y' is provided:
    1. scores:          model's classification scores
    2. acetylGenesPred: list of genes/reactions predicted to be Acetyl
    3. phosGenesPred:   list of genes/reactions predicted to be Phos
    4. ypred:           array of model class predictions as integers 
    
    *** If 'y' is NOT provided:
    1. acetylGenesPred: list of genes/reactions predicted to be Acetyl
    2. phosGenesPred:   list of genes/reactions predicted to be Phos
    3. ypred:           array of model class predictions as integers 

    '''
    
    # make folder for output figures
    path = '../figures/mdl_predict/'
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of directory %s failed (may already exist)" % path)
        
    # set font for figures
    pltFont = {'fontname':'Arial'}
    
    # define feature names and # classes
    feat_names = X.columns
    
    # reset indices
    X = X.reset_index(drop=True)
    gene_reactions = gene_reactions.reset_index(drop=True)  
    
    # get model predictions and probabilities
    ypred = mdl.predict(X)
    print(Counter(ypred))
    yproba = mdl.predict_proba(X)
    
    if class_names is None:
        class_names=[]
        for cl in np.sort(pd.Series(ypred).unique()):
            class_names.append(np.array2string(cl))
    print()
        
    num_class = len(class_names)

    # if y is given, perform following analyses of
    # model's classification performance:
    if y is not None:
        
        # transform Target classes to 0, 1, 2
        y = y.reset_index(drop=True)
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = pd.Series(le.transform(y))
        
        cm = confusion_matrix(y, ypred)
        
        path2 = '../results'
        try:
            os.makedirs(path2)
        except OSError:
            print("Creation of directory %s failed (may already exist)" % path2)
            
        if num_class > 2:
            print("MULTI-CLASS PROBLEM!")
            # write classification results to CSV
            for class_num, class_name in zip([0, 2], ["Phosphorylation", "Acetylation"]):
                
                genes_FP = gene_reactions.iloc[np.where((y != class_num) & (ypred == class_num))]
                if len(genes_FP)>0:
                    genes_FP.to_csv('../results/{}_{}_FalsePos_GeneRxns.csv'.format(condition,class_name),
                                 index=False)  
                    
                genes_FN = gene_reactions.iloc[np.where((y == class_num) & (ypred != class_num))]
                if len(genes_FN)>0:
                    genes_FN.to_csv('../results/{}_{}_FalseNeg_GeneRxns.csv'.format(condition,class_name),
                                 index=False)
                    
                genes_TP = gene_reactions.iloc[np.where((y == class_num) & (ypred == class_num))]
                if len(genes_TP)>0:
                    genes_TP.to_csv('../results/{}_{}_TruePos_GeneRxns.csv'.format(condition,class_name),
                                 index=False)
                
                # calculate classification scores based on number of classes
                TP = cm[0,0]
                FP = cm[1,0]
                TN = sum(cm[1,1:]) 
                FN = sum(cm[0,1:])  
                accuracy = accuracy_score(y, ypred)
                precision = TP/(TP+FP)
                recall = TP/(TP+FN)
                f1 = 2*(recall*precision)/(recall+precision)
                mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                r = np.corrcoef(y, ypred)[0, 1]
        else: 
            print("BINARY PROBLEM!")
            # write classification results to CSV
            genes_FP = gene_reactions.iloc[np.where((y != 1) & (ypred == 1))]
            if len(genes_FP)>0:
                genes_FP.to_csv('../results/{}_{}_FalsePos_GeneRxns.csv'.format(condition,class_names[1]),
                             index=False)  
                
            genes_FN = gene_reactions.iloc[np.where((y == 1) & (ypred != 1))]
            if len(genes_FN)>0:
                genes_FN.to_csv('../results/{}_{}_FalseNeg_GeneRxns.csv'.format(condition,class_names[1]),
                             index=False)
                
            genes_TP = gene_reactions.iloc[np.where((y == 1) & (ypred == 1))]
            if len(genes_TP)>0:
                genes_TP.to_csv('../results/{}_{}_TruePos_GeneRxns.csv'.format(condition,class_names[1]),
                             index=False)
            # calculate classification scores
            accuracy = accuracy_score(y, ypred)
            f1 = f1_score(y, ypred, average="binary",pos_label=pos_label)
            recall = recall_score(y, ypred, average="binary",pos_label=pos_label)
            precision = precision_score(y, ypred, average="binary",pos_label=pos_label)
            mcc = matthews_corrcoef(y, ypred)
            r = np.corrcoef(y, ypred)[0, 1] 
            
            
        scores = [accuracy, recall, precision, f1 , mcc, r]
        score_names = ['Accuracy', 'Recall', 'Precision', 'F1', 'MCC', 'R']
        df_scores = pd.DataFrame(data=scores, index=score_names)
        
        # confusion matrix figure
        if confusion_mat:
            ax = make_confusion_matrix(y, ypred, figsize = (8,6),
                                  categories = class_names,
                                  xyplotlabels = True,
                                  sum_stats = False,
                                  cbar = False)
            # ax.yaxis.label.set_fontsize(24)
            plt.ylabel("Experimental Labels", fontsize=24)
            plt.xlabel("Predicted Labels", fontsize=24)
            plt.savefig(path+"{}_confusionMat.png".format(condition),
                     bbox_inches='tight', dpi=600)
            
        # score bar graph
        if bar is True:
            plt.rcParams.update(plt.rcParamsDefault) 
            fig, ax = plt.subplots(figsize=(8,6))

            ax.bar(score_names, scores,
                   align='center',
                   alpha=0.5,
                   ecolor='black',
                   capsize=10,
                   width=0.8)
            ax.set_ylim([0, 1.0])
            ax.set_xticklabels(score_names,
                                rotation=45, ha="right",
                                rotation_mode="anchor")
            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.yaxis.grid()
            plt.tight_layout()
            plt.savefig(path+"{}_ScoresBarGraph.png".format(condition),
                     bbox_inches='tight', dpi=600)


        if pairwise is True and num_class>2:
            # Sort feature importances in descending order
            indices = np.argsort(mdl.feature_importances_)[::-1]
            # Rearrange feature names 
            names = [feat_names[i] for i in indices]   
            
            for class_num, class_name in zip([0, 2], ["Phos", "Acetyl"]):
                if sum(y==class_num)>0:
                    df_deg = X.loc[y==class_num]             # get X for DEG rows
                    y2 = y[y==class_num]                     # get y for Phos rows
                    ypred2 = ypred[y==class_num]             # get ypred for Phos rows
                    df_deg['Correct'] = y2 == ypred2         # get classification results
                    
                    # get 5 top feats
                    impFeats = list(names[0:5])
                    impFeats.append("Correct")
                    print("Num correct: {} ".format(sum(df_deg.Correct)))
    
                    # create pairwise plot for Phos rows
                    sns.set(font_scale=2)  
                    plt.figure(figsize=(8,6))
                    sns.pairplot(df_deg[impFeats],
                                 hue="Correct", hue_order = [False,True],
                                 plot_kws={'s':70})
                    plt.savefig(path+"{}_pairwise_Predict{}.png".format(condition, class_name),
                             bbox_inches='tight', dpi=600)
                    plt.show()
        
        if swarm is True and num_class>2:
            for class_num, class_name in zip([0, 2], ["Phos", "Acetyl"]):
                if sum(y==class_num)>0:
                    df_deg = X.loc[y==class_num]             # get X for DEG rows
                    y2 = y[y==class_num]                     # get y for Phos rows
                    ypred2 = ypred[y==class_num]             # get ypred for Phos rows
                    df_deg['Correct'] = y2 == ypred2         # get classification results
                   
                    numeric_feats = df_deg.select_dtypes(include='float64').columns
                    
                    sns.set()
                    f, axs = plt.subplots(3,4, figsize=(8,6))
                    for i, ax in enumerate(axs.reshape(-1)): 
                        sns.swarmplot(x="Correct", y=numeric_feats[i], 
                                      data=df_deg, ax=ax,
                                      order = [False,True], size=3)
                        ax.set_xlabel('')   # remove x-axis label
                        ax.set_ylabel(numeric_feats[i], fontsize=16, **pltFont)
                        ax.set_xticklabels(["FN", "TP"])
                        ax.tick_params(axis='both', which='major',
                                       labelsize=16)
                    plt.tight_layout()
                    plt.savefig(path+"{}_SwarmPlot_{}True.png".format(condition, class_name),
                             bbox_inches='tight', dpi=600)
                    plt.show()
            
            
        if boxplot is True and num_class>2:
            for class_num, class_name in zip([0, 2], ["Phos", "Acetyl"]):
                if sum(y==class_num)>0:
                    
                    # assign line colors based on classification
                    classification_groups = np.where((y==class_num) & (ypred==class_num),'TP', # TP
                        np.where((y!=class_num)&(ypred!=class_num),'TN', # TN
                        np.where((y==class_num) & (ypred!=class_num),'FN', # FN
                                          'FP')))  # FP
                    X_temp = X.copy()
                    X_temp['classification'] = classification_groups

                    numeric_feats = X.select_dtypes(include='float64').columns
                    sns.set()
                    labels = ["TP","FP","TN","FN"]
            
                    f, axs = plt.subplots(3,4, figsize=(8,6))
                    for i, ax in enumerate(axs.reshape(-1)):
                        sns.boxplot(y=numeric_feats[i], x="classification", data=X_temp,
                                    ax=ax, order=labels)
                        # ax.set_ylabel(str(i))
                        ax.set_xlabel('')   # remove x-axis label
                        ax.set_ylabel(numeric_feats[i], fontsize=16, **pltFont)
                        ax.set_xticklabels(labels,rotation=45, ha="center")
                        ax.tick_params(axis='both', which='major',
                                       labelsize=14)
                    plt.tight_layout()
                    plt.savefig(path+"{}_Boxplot_{}True.png".format(condition, class_name),
                             bbox_inches='tight', dpi=600)
                    plt.show()
            
            
        if explainer and num_class>2:
            
            expected_value = explainer.expected_value
            
            ### decision plot for ~50 random observations            
            for class_num, class_name in zip([0, 2], ["Phosphorylation", "Acetylation"]):
                print("CURRENT CLASS: {}, COUNT = {}".format(class_name,sum(y==class_num)))
                
                if sum(y==class_num)>0:

                    yclass = np.where((y==class_num) & (ypred==class_num),'TP', # TP
                        np.where((y!=class_num)&(ypred!=class_num),'TN', # TN
                        np.where((y==class_num) & (ypred!=class_num),'FN', # FN
                                          'FP')))  # FP
                    
                    
                    # get features for 50 samples and find those misclassified
                    X_temp = X.copy()
                    X_temp["classification"] = yclass
                    X_temp2 = X_temp.groupby("classification").sample(n=10, random_state=123)
                    
                    # assign line colors based on classification
                    ycol = np.where(X_temp2.classification=="TP",'blue', # TP
                        np.where(X_temp2.classification=="TN",'red', # TN
                        np.where(X_temp2.classification=="FN",'darkred', # FN
                                          'deepskyblue')))  # FP

                    features = X_temp2[feat_names]
                    misclass = (X_temp2.classification=="FP")|(X_temp2.classification=="FN")
        
                    # get shap values of specified observations
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        shap_values_decision = explainer.shap_values(features)[class_num]   
                        shap_interaction_values = explainer.shap_interaction_values(features)[class_num]
                    if isinstance(shap_interaction_values, list):
                        shap_interaction_values = shap_interaction_values
                
                    # create decision plot
                    plt.rcParams.update(plt.rcParamsDefault) 
                    f1 = plt.figure()
                    shap.decision_plot(expected_value[class_num], shap_values_decision, #shap_interaction_values
                                       features,
                                       highlight = misclass,
                                       color_bar=False,
                                       legend_labels=None,
                                       feature_order='hclust',
                                       show = False)
                    plt.title("Decision Plot: {}".format(class_name),
                          fontsize=20, **pltFont)  
                    
                    # fix line colors and weights
                    num_lines=len(plt.gca().lines)
                    for line, color in zip(plt.gca().lines[len(feat_names):num_lines],
                                           ycol):
                        line.set_linewidth(2)
                        line.set_color(color)                    
                    
                    # create legend
                    colors = ['blue', 'red', 'darkred','deepskyblue']
                    styles = ['-','-','--','--']
                    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle=s) for c,s in zip(colors, styles)]
                    labels = ['TP', 'TN', 'FN','FP']
                    # save fig
                    plt.legend(lines, labels, 
                               fontsize=15, loc='lower right')
                    plt.savefig(path+"{}_SHAPDecisionPlot_{}.png".format(condition, class_name),
                             bbox_inches='tight', dpi=600)
                    plt.show()
                        
                    # loop through phos and acetyl classes
                    for class_group in (["TP","FP"]):
                        # get false positive/negative genes
                        X_group= X_temp[X_temp.classification==class_group][feat_names]
                       
                        if class_group == "TP":
                            title = "True Positives"
                        elif class_group == "FP":
                            title = "False Positives"
                            
                        # get shap values for FP genes
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            shap_values_decision = explainer.shap_values(X_group)[class_num]   
                            shap_interaction_values = explainer.shap_interaction_values(X_group)[class_num]
                        if isinstance(shap_interaction_values, list):
                            shap_interaction_values = shap_interaction_values
                    
                        ## decision plot for FP genes
                        plt.rcParams.update(plt.rcParamsDefault) 
                        f1 = plt.figure()
                        shap.decision_plot(expected_value[class_num], shap_values_decision, #shap_interaction_values
                                           X_group,
                                           color_bar=True,
                                           legend_labels=None,
                                           feature_order='hclust',
                                           show = False)
                        plt.title("Decision Plot: {} {}".format(class_name, title),
                                  fontsize=20, **pltFont) 
                        plt.yticks(fontsize=18, **pltFont)
                        plt.xticks(fontsize=18, **pltFont)
                        plt.savefig(path+"{}_ShapDecisionPlot_{}_{}.png".format(
                            condition,class_name, class_group),
                            bbox_inches='tight', dpi=600)
                        plt.show()
                        
                        ## summary plot for FP genes
                        shap_values = explainer.shap_values(X_group)
                        plt.figure()
                        shap.summary_plot(shap_values[class_num], X_group,
                                          color_bar=False,
                                          show=False)
                        plt.title("Summary Plot: {} {}".format(class_name, title),
                                  fontsize=20, **pltFont) 
                        plt.yticks(fontsize=18, **pltFont)
                        plt.xticks(fontsize=18, **pltFont)
                        plt.xlabel("SHAP Value (impact on model output)",fontsize=20, **pltFont)
                        
                        # make our own color bar so that we can adjust font/label sizes
                        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["#4d73ff","#ff1303"])
                        cnorm = mcol.Normalize(vmin=0,vmax=1)
                        m = pltcm.ScalarMappable(norm=cnorm,cmap=cm1)
                        m.set_array([0, 1])
                        cb = plt.colorbar(m, ticks=[0, 1], aspect=1000)
                        cb.set_ticklabels(["Low", "High"])
                        cb.set_label("Feature Value", size=18, labelpad=-10)
                        cb.ax.tick_params(labelsize=18, length=0)
                        cb.set_alpha(1)
                        cb.outline.set_visible(False)
                        bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
                        cb.ax.set_aspect((bbox.height - 0.9) * 20)
                        
                        plt.savefig(path+"{}_ShapSummaryPlot_{}_{}.png".format(
                            condition, class_name, class_group),
                            bbox_inches='tight', dpi=600)
                        plt.show()                         
                        
        # get gene predictions for output
        if num_class > 2:
            acetylGenesPred = gene_reactions[ypred==2]
            phosGenesPred = gene_reactions[ypred==0]
            print("SCORES: ", df_scores)
            return scores, acetylGenesPred, phosGenesPred, ypred
        else:
            GenesPred = gene_reactions[ypred==1]
            print("SCORES: ", df_scores)
            return scores, GenesPred, ypred
        
    else:   # no y_true given (assume that classes are unknown)
        
        print("# of Predicted Phos. Genes-Rxn Pairs: {}".format(Counter(ypred)[0]))
        print("# of Predicted Unreg. Genes-Rxn Pairs: {}".format(Counter(ypred)[1]))
        print("# of Predicted Acetyl. Genes-Rxn Pairs: {}".format(Counter(ypred)[2]))


        ## create new dataset with features, gene names and predicted classes
        # reset indices         
        df_new = X.copy()
        df_new["genes"] = gene_reactions.genes
        df_new["reaction"] = gene_reactions.reaction
        df_new["ypred"] = ypred
        
        acetylGenesPred = gene_reactions[ypred==2]
        phosGenesPred = gene_reactions[ypred==0]
        
        # output gene predictions to CSV        
        numeric_feats = X.select_dtypes(include='float64').columns
        pltFont = {'fontname':'Arial'}
        
        y_classes = np.where(ypred==0,'Ph',
        np.where(ypred==1,'Un','Ac'))
         
        sns.set()
        f, axs = plt.subplots(3,4, figsize=(8,6))
        for i, ax in enumerate(axs.reshape(-1)):
            sns.boxplot(y=df_new[numeric_feats[i]], x=y_classes,
                        ax=ax, order=["Ph","Un","Ac"])
            ax.set_xlabel('')   # remove x-axis label
            ax.set_ylabel(numeric_feats[i], fontsize=16, **pltFont)
            ax.tick_params(axis='both', which='major',
                            labelsize=16)
        plt.tight_layout()
        plt.savefig(path+"{}_predictionsBoxplot.png".format(condition),
                  bbox_inches='tight', dpi=600)
        plt.show()  
        
        if gene_reactions.any:
            for class_num, class_name in zip([0, 2], ["Phos", "Acetyl"]):
                DEGs = gene_reactions.iloc[ypred==class_num]
                if len(DEGs)>0:
                   DEGs.to_csv('../results/{}_Predicted{}Genes.csv'.format(condition,class_name),
                                 index=False)
    
        return acetylGenesPred, phosGenesPred, ypred
            
    