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
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import math
from sklearn.model_selection import GridSearchCV
import shap
from collections import Counter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.lines import Line2D
import matplotlib.colors as mcol
import matplotlib.cm as pltcm


shap.initjs()



def corr_heatmap(df, condition):
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(10,10))
    # plot heat map
    plt.figure
    sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn",
                     vmin=-0.75, vmax=0.75,fmt='.2f')
    plt.savefig("../figures/correlation_heatmaps/{}_CorrHeatmap.png".format(condition),
                     bbox_inches='tight', dpi=600)
    
def prune(tree):
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
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
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
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
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

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
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

    if xyplotlabels :  # show True/Predicted labels and put stats below plot
        plt.ylabel('True label', fontweight='bold', **hfont)
        plt.xlabel('Predicted label' + stats_text, fontweight='bold', **hfont)
    elif cbar:   # show color bar on right and stats below 
        plt.xlabel(stats_text, fontsize=15, **hfont)
    else:   # no color or True/Predicted labels, so put stats on right
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_label_coords(1.25,0.75)
        plt.ylabel(stats_text, fontsize=18, rotation=0, **hfont)#labelpad=75
    
    if title:
        plt.title(title, **hfont)
        
    plt.tight_layout()
    return ax


def xgb_func(X, y, num_iter, condition, class_names=None, depth="deep",
             imbalance="none"):
    
    # define font type for plots
    pltFont = {'fontname':'Arial'}
    
    # define feature names
    feat_names = X.columns
    
    # transform Target classes to 0, 1, 2
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = pd.Series(le.transform(y))
    
    if class_names is None:
        class_names=[]
        for cl in y.unique():
            class_names.append(np.array2string(cl))
    
    # get number of classes
    num_class = len(np.unique(y))
    print("Number of class: {}".format(num_class))
   
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

 
    # Prepare data and classifiers, based on binary vs multi-class problem
    if num_class == 2:

        # define classifier and hyperparameter tuning
        classifier = xgboost.XGBClassifier(objective='binary:logistic',
                                           n_estimators=200,
                                           random_state=123)
    
        random_search = RandomizedSearchCV(classifier, param_distributions=params,
                                          n_iter=10, scoring='f1',  # 100
                                          n_jobs=-1, cv=3, verbose=3,
                                          random_state=123) 
        avg = "binary"
    
    elif num_class > 2:
        # define classifier and hyperparameter tuning
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
        
    # StratifiedKfold method
    cv = StratifiedKFold(n_splits=num_iter,
                         shuffle=True, 
                         random_state=123)
    
    # create empty lists to store CV scores
    acc_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    mcc_list = []
    auc_list = []
    r_list = []
    
    y_test = []
    y_pred = []
    # cmCV = [[0,0,0],[0,0,0],[0,0,0]]
    cmCV = np.zeros((num_class, num_class))
    paramDict = {}

    count = 0
    
    for train_index, test_index in cv.split(X, y):
        X_trainCV, X_testCV = X.iloc[train_index], X.iloc[test_index]
        y_trainCV, y_testCV = y.iloc[train_index], y.iloc[test_index]
            
        
        if isinstance(imbalance, (list, float)):
            ## transform the dataset w/ SMOTE
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
            
        elif imbalance=="smote":
            ## transform the dataset w/ SMOTE
            oversample = SMOTE()
            X_trainCV, y_trainCV = oversample.fit_resample(X_trainCV, y_trainCV)
            random_search.fit(X_trainCV, y_trainCV)
            
        elif imbalance=="adasyn":
            class_values = y_trainCV.value_counts()
            smote_dict = {0:int(round(class_values[0]*0.75)),
                          1:class_values[0],
                          2:int(round(class_values[0]*0.75))}
            ada = ADASYN(sampling_strategy = smote_dict,
            random_state=123, n_neighbors=10)
            X_trainCV, y_trainCV = ada.fit_resample(X_trainCV,y_trainCV)

            random_search.fit(X_trainCV, y_trainCV)

        elif imbalance=="undersample":
            ## transform the dataset w/ SMOTE
            nr = NearMiss() 
            X_trainCV, y_trainCV = nr.fit_sample(X_trainCV, y_trainCV)
            random_search.fit(X_trainCV, y_trainCV)
        
        elif imbalance=="none":
            random_search.fit(X_trainCV, y_trainCV)
            
        elif imbalance=="balanced":            
            weights = class_weight.compute_sample_weight("balanced", y_trainCV)
            
            random_search.fit(X_trainCV, y_trainCV,
                              sample_weight=weights)
        
        randomSearch_mdl = random_search.best_estimator_
            
        # tune gamma
        params_gamma = {'gamma':[0, 0.1, 0.3, 0.5]}
        gamma_search = GridSearchCV(estimator = randomSearch_mdl, 
                            param_grid = params_gamma, scoring='f1_macro',
                            n_jobs=-1 , cv=3)
    
        gamma_search.fit(X_trainCV, y_trainCV)
                
        
        best_Mdl = gamma_search.best_estimator_
        print("Cross-val Fold {}, Model Params: {}".format(count, best_Mdl))
        paramDict[count] = best_Mdl.get_params
        y_predCV = best_Mdl.predict(X_testCV)
        # y_probaCV = best_Mdl.predict_proba(X_testCV)
        
        y_test.extend(y_testCV)
        y_pred.extend(y_predCV)
        
        cm = confusion_matrix(y_testCV, y_predCV)
        print("current cm: \n",cm)
        cmCV = cmCV+cm
        print("Combined cm: \n", cmCV)
        
        accuracy = accuracy_score(y_testCV, y_predCV)
        f1 = f1_score(y_testCV, y_predCV, average=avg)
        recall = recall_score(y_testCV, y_predCV, average=avg)
        precision = precision_score(y_testCV, y_predCV, average=avg)
        mcc = matthews_corrcoef(y_testCV, y_predCV)
        r = np.corrcoef(y_testCV, y_predCV)[0, 1]
        # if num_class>2:
        #     roc_auc = roc_auc_score(y_testCV, y_probaCV,
        #                           multi_class="ovo", average=avg)
        # else:
        #     roc_auc = roc_auc_score(y_testCV, y_probaCV[:, 1])
            
        # assign scores to list
        acc_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        mcc_list.append(mcc)
        # auc_list.append(roc_auc)
        r_list.append(r)
    
        count = count+1
             
    # print final CM
    print("final CV confusion matrix: \n",cmCV)
   
    ### plot confusion matrix results 
    # create folder for CV figure
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
    # AUC = np.mean(auc_list)
    Corr = np.mean(r_list)
    
    scores = [Accuracy, Recall, Precision, F1, MCC, Corr] #AUC,
    
    # get stats for CV scores
    loop_scores = {'Accuracy':acc_list,
                   'Recall':recall_list,
                   'Precision':precision_list,
                   'F1':f1_list,
                   'MCC':mcc_list,
                   'R':r_list}
                    #    'AUC':auc_list,

                    
    df_loop_scores = pd.DataFrame(loop_scores)
    print("Model score statistics: ")
    loop_stats = df_loop_scores.describe()
    print(loop_stats)
    
    CIs = st.t.interval(alpha=0.95, df=len(df_loop_scores)-1,
              loc=np.mean(df_loop_scores), scale=st.sem(df_loop_scores))

    # plot CV scores
    plt.rcParams.update(plt.rcParamsDefault) 
    plt.rcParams['xtick.major.pad']='10'
    fig, ax = plt.subplots(figsize=(8,6))
    # ax.set_facecolor('0.9')
    ax.bar(df_loop_scores.columns, scores,
           yerr=loop_stats.loc['std',:],
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10,
           width=0.8)
    # ax.set_ylabel('Score', fontsize=14)
    ax.set_ylim([0, 1.0])
    plt.yticks(**pltFont)
    ax.set_xticks(df_loop_scores.columns)
    ax.set_xticklabels(df_loop_scores.columns,**pltFont,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis='both', which='major', labelsize=24)
    # ax.set_title('XGBoost Cross-Validation')
    ax.yaxis.grid(True)
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('../figures/crossval/{}_XGB_crossVal_barGraph.png'.format(condition),
                bbox_inches='tight', dpi=600)
    plt.show()


    # create dataframe with mean scores
    data = {'Metric':['Acc', 'Recall', 'Precision','F1', 'MCC', 'PearsonsR'], 
      'Scores':[Accuracy, Recall, Precision, F1, MCC, Corr]} 
    df_scores = pd.DataFrame(data)
    df_scores = df_scores.set_index('Metric')



    ### train model on entire training dataset using params 
    ### from best CV model
    
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
        
    elif imbalance=="smote":
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


     # Calculate feature importances
    importances = final_Mdl.feature_importances_
    
     # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
     # Rearrange feature names so they match the sorted feature importances
    feat_names = X.columns
    names = [feat_names[i] for i in indices]   # for sfs
    
     # Create plot
    fig1 = plt.figure()
     # Create plot title
    plt.title("XGBoost Feature Importance")
     # Add bars
    plt.bar(range(X.shape[1]), importances[indices])  
     # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names,
                fontsize=18, rotation=45, horizontalalignment="right")
    plt.yticks(fontsize=20)
    plt.bar(range(X.shape[1]), importances[indices])  
    fig1.savefig("../figures/crossval/{}_XGB_featureImps.png".format(condition),
                  bbox_inches='tight', dpi=600)
    # Show plot
    plt.show()
    
    return final_Mdl, loop_stats

    

 ##### SHAP ANALYSIS #####
 
def shap_func(xgbModel, X, condition, class_names):
    
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

    # define feature names and abbreviated feature names
    feat_names = X.columns
    feats_short = ['geneKO','maxATP','growthAC','close','degree','between',
               'pageRank','reverse','rawVmin','rawVmax','PFBA','Kcat',
               'MW']
    
    # if no background given, use tree_path_dependent method
    explainer = shap.TreeExplainer(xgbModel,
               feature_perturbation='tree_path_dependent') 
    X_test = X
    shap_values = explainer.shap_values(X_test)   # X_test
    
    ## shap summary and dependence plots ##
    
    # combined
    num_class = len(class_names)
    if num_class > 2:
        if class_names is None:
            class_names = xgbModel.classes_
        # combined summary 
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
             
            # dependence plots (only plot for top 3 feats)
            vals = np.abs(shap_values[which_class]).mean(0)
            # Sort feature importances in descending order
            indices = np.argsort(vals)[::-1]
            
            # Rearrange feature names so they match the sorted feature importances
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
         fig2 = plt.figure()
         shap.summary_plot(shap_values, X_test, show=False) #class_names = multi_classes
         plt.title("SHAP Summary Plot: {}".format(condition),fontsize=15)
         plt.show()
         fig2.savefig("../figures/SHAP/{}_SummaryPlot.png".format(condition),
                      bbox_inches='tight', dpi=600)
   
         shap.dependence_plot("rank(0)", shap_values, X_test, show=False) # *****
         plt.title("Dependence Plot",fontsize=15)
         plt.savefig("../figures/SHAP/dependence_plots/{}_DependencePlot.png".format(condition),
                     dpi=600)
         plt.show()
      

    # get expected value
    expected_value = explainer.expected_value
    print(expected_value)
    
    ### shapley heatmap
    if num_class == 2:
        explainer_shaps = explainer(X_test)    # X_test  
        shap.plots.heatmap(explainer_shaps,
                           max_display=len(feat_names))
    elif num_class > 2:
        multi_heatmap(explainer, X_test, num_class, order="output",
                      class_names=class_names,
                      condition=condition,
                      max_display=len(feat_names))
          
    return explainer, shap_values #, CVscore


def multi_heatmap(explainer, X, num_class, order="explanation",
                  feat_values=None,cmap="bwr", class_names=None,
                  max_display=10, condition=None, save=True,
                  showOutput=True):
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
        fig = plt.figure()
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


def predict_genes(X, y, all_genes, select_genes, xgb_clf,
                  explainer, class_names, condition="PredictGenes"):
    
    path= "../figures/SHAP/predictGenes"
    try:
        os.mkdir(path)
    except OSError:
        print ("Directory %s failed (may already exist)" % path)
    else:
        print ("Successfully created the directory %s " % path)
    pltFont = {'fontname':'Arial'}

    feature_names = X.columns
    feat_names_list = feature_names.tolist()
    num_class = len(y.unique())
    
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = pd.Series(le.transform(y))
    
    # get indices of gene of interest

    i_genes = []
    for gene in select_genes:
        bool_list = all_genes["genes"].eq(gene)
        i_gene = list(compress(range(len(bool_list)), bool_list))
        i_genes.extend(i_gene)
    
        
    # # create training and test sets
    # X_train =  X.loc[~X.index.isin(i_genes), feature_names]
    X_test = X.loc[X.index.isin(i_genes), feature_names]
    
    # y_train = y[~y.index.isin(i_genes)]
    y_test = y[y.index.isin(i_genes)]
    
    # get model prediction
    y_pred = xgb_clf.predict(X_test) 
    
    # show confusion matrix
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


    ### shapley analysis ###

    ## decision plot ##
    expected_value = explainer.expected_value
    print(expected_value)
    
    for gene in select_genes:
        
        bool_genes = df["Test Genes"]== gene   # get rows for current gene
        
        features=X_test[bool_genes]   # get features for current gene
        features_display=X_test[bool_genes]
        
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
        
        # will create decision plot for each class, so loop through classes
        for j in y.unique()[::-1]:
            class_genes = y_test_select == j   # ID genes of current class in y_test (to be used for highlighting)
        
        # get shap values of specified observations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values_decision = explainer.shap_values(features)[j]   # j starts with -1, so add 1 to get index 0
                shap_interaction_values = explainer.shap_interaction_values(features)[j]
            if isinstance(shap_interaction_values, list):
                shap_interaction_values = shap_interaction_values
    
    # tkt_genes = df_DT["test_genes"]=="TKT"
    
            # create decision plot
            f1 = plt.figure()
            shap.decision_plot(expected_value[j], shap_values_decision,
                               features,
                               highlight=class_genes,
                               show=False)
            plt.title("SHAP Decision Plot: {}, {}".format(gene, class_names[j]), fontsize=15)
            plt.savefig("../figures/SHAP/predictGenes/{}_{}_{}DecisionPlot.png".format(condition, gene, class_names[j]),
                        bbox_inches='tight', dpi=600)
            plt.show()
                
            
            
    ### multi-class decision plot ###

    for gene in select_genes:   # loop through select_genes

        # function create legend labels using class name and log odds value
        def class_labels(row_index):
            return [f'{j} ({logodds[row_index, i].round(2):.2f})' for i,j in enumerate(class_names)]
             
    
        bool_genes = (df["Test Genes"] == gene) & (df["y_test"]==df["y_pred"])   # ID rows with current gene and correct classification
        
        # if at leat 2 correct predictions
        if sum(bool_genes) > 0:
            if sum(bool_genes) == 1:
                row_index = 1   # explain 1st two observations for each gene
            else:
                row_index = 2
        
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
            for row in range(row_index):
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
                plt.savefig("../figures/SHAP/predictGenes/{}_{}{}_{}_MultiOutputPlot.png".format(
                    condition, gene, row, regType),
                    bbox_inches='tight', dpi=600)        
                plt.show()
                


def decisionTree(X, y, class_names, weighting="balanced", weights=None, pruneLevel=0, condition="Condition", save=True):
    
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
                
                # create array of weights
                # class_weights = list(weight.values())
                # w_array = np.ones(y.shape[0], dtype = 'float')
                # for i, val in enumerate(y):
                #     w_array[i] = class_weights[val+1]
                    
                # Fit it to the data
                tree_cv.fit(X,y) #,sample_weight=w_array
                treeDict_VarWeights[count] = tree_cv
                scores.append(tree_cv.best_score_)
                print("Best score is {}".format(tree_cv.best_score_))
        
        
            maxpos = scores.index(max(scores)) 
            print(maxpos+1)
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
        
        ax1 = plt.subplot2grid((4, 4), (0, 3), colspan=1,rowspan=1)
        sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                            annot=True, cmap='Blues', fmt='d')
        
        ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
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
    
            
        
def mdl_predict(mdl, X, condition, class_names, gene_reactions,
                y=None, confusion_mat=True,
                bar=False, swarm=False,
                pairwise=False, boxplot=False,
                explainer=None, gscatter=False):
    
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
    num_class = len(class_names)
    
    # reset indices
    X = X.reset_index(drop=True)
    gene_reactions = gene_reactions.reset_index(drop=True)  
    
    # get model predictions and probabilities
    ypred = mdl.predict(X)
    yproba = mdl.predict_proba(X)

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
        if num_class > 2:
            TP = cm[0,0]
            FP = cm[1,0]
            TN = sum(cm[1,1:]) #+cm[1,2]
            FN = sum(cm[0,1:])  #+cm[0,2]
            accuracy = accuracy_score(y, ypred)
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            f1 = 2*(recall*precision)/(recall+precision)
            mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
            r = np.corrcoef(y, ypred)[0, 1]
        else: 
            accuracy = accuracy_score(y, ypred)
            f1 = f1_score(y, ypred, average="binary")
            recall = recall_score(y, ypred, average="binary")
            precision = precision_score(y, ypred, average="binary")
            mcc = matthews_corrcoef(y, ypred)
            r = np.corrcoef(y, ypred)[0, 1] 
            
        scores = [accuracy, recall, precision, f1 , mcc, r]
        score_names = ['Accuracy', 'Recall', 'Precision', 'F1', 'MCC', 'R']
        
        
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
            ax.yaxis.set_minor_locator(AutoMinorLocator())       
            plt.tight_layout()
            plt.savefig(path+"{}_ScoresBarGraph.png".format(condition),
                     bbox_inches='tight', dpi=600)


        if pairwise is True:
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
        
        if swarm is True:
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
            
            
        if boxplot is True:
            for class_num, class_name in zip([0, 2], ["Phos", "Acetyl"]):
                if sum(y==class_num)>0:
                    
                    # assign line colors based on classification
                    classification_groups = np.where((y==class_num) & (ypred==class_num),'TP', # TP
                        np.where((y!=class_num)&(ypred!=class_num),'TN', # TN
                        np.where((y==class_num) & (ypred!=class_num),'FN', # FN
                                          'FP')))  # FP
                    X_temp = X.copy()
                    X_temp['classification'] = classification_groups
                    
                    # df_deg = X.loc[y==class_num]             # get X for DEG rows
                    # y2 = y[y==class_num]                     # get y for Phos rows
                    # ypred2 = ypred[y==class_num]             # get ypred for Phos rows
                    # df_deg['Correct'] = y2 == ypred2         # get classification results
                   
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
            
            
        if explainer:

            expected_value = explainer.expected_value
            
            ### decision plot for 50 random observations
            
            # use undersampling so even number of Phos and Unreg

            
            # sample 50 observations from undersampled DF
            
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
            
            
        if gscatter:
            # get y and y_proba values where ypred==-1
            df_phosPred = pd.DataFrame(X[ypred==0])
            numeric_feats = df_phosPred.select_dtypes(include='float64').columns

            df_phosPred["Target"] = pd.DataFrame(y[ypred==0])
            df_phosPred["Probability"] = yproba[ypred==0][:,0]
            # df_phosPred["Row"] = range(0,len(df_phosPred))
            df_phosPred = df_phosPred.assign(Classification=df_phosPred.Target.map({0: "TP", 1: "FP"}))
            print(df_phosPred.head())
            print("Number of predicted Phos: {}".format(len(df_phosPred)))

            sns.set()
            f, axs = plt.subplots(3,4, figsize=(10,8))
            for i, ax in enumerate(axs.reshape(-1)):
                sns.scatterplot(data=df_phosPred, x=numeric_feats[i], y="Probability",
                     ax=ax, hue="Classification",
                     palette=["green","red"], legend=False) 
                ax.tick_params(axis='both', which='major',
                               labelsize=16)
                ax.set_ylabel("Probability", fontsize=16, **pltFont)
                ax.set_xlabel(numeric_feats[i], fontsize=16, **pltFont)

            legend_elements = [Line2D([0], [0], marker='o', color='w', label='TP',
                                      markerfacecolor='green', markersize=15,
                                      linewidth=0),
                              Line2D([0], [0], marker='o', color='w', label='FP',
                                     markerfacecolor='red', markersize=15,
                                     linewidth=0)]
            f.legend(handles=legend_elements, loc='upper right',
                     title='Classification', bbox_to_anchor=(1.1, 1))
            
            plt.tight_layout()
            plt.savefig(path+"{}_scatterPlot_PhosPred.png".format(condition),
                        bbox_inches='tight', dpi=600)
            plt.show()
            
        # get phos and acetyl genes for output
        acetylGenesPred = gene_reactions[ypred==2]
        phosGenesPred = gene_reactions[ypred==0]
            
        return scores, acetylGenesPred, phosGenesPred, ypred
        
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

                
        path = '../figures/predictUnknown/'
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of directory %s failed (may already exist)" % path)
            
        numeric_feats = X.select_dtypes(include='float64').columns
        pltFont = {'fontname':'Arial'}
        
        y_classes = np.where(ypred==0,'Ph',
        np.where(ypred==1,'Un','Ac'))
         
        sns.set()
        f, axs = plt.subplots(3,4, figsize=(8,6))
        for i, ax in enumerate(axs.reshape(-1)):
            sns.boxplot(y=df_new[numeric_feats[i]], x=y_classes,
                        ax=ax, order=["Ph","Un","Ac"])
            # ax.set_ylabel(str(i))
            ax.set_xlabel('')   # remove x-axis label
            ax.set_ylabel(numeric_feats[i], fontsize=16, **pltFont)
            # ax.set_xticklabels(["Ph","Un","Ac"])
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
    
        return df_new, phosGenesPred, acetylGenesPred, ypred
            
        

    