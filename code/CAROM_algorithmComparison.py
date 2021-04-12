#!/usr/bin/env python
# coding: utf-8

"""
'CAROM_algorithmComparison.py' performs a preliminary analysis of
several ML algorithms for predicting post-translational modifications
in the metabolic network, as described in the CAROM manuscript.

The superior performance of XGBoost demonstrated in this script supported
our decision to use this algorithm for the primary model. Several other models
trained here are presented in the supplementary figures.

@author: Kirk Smith
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             matthews_corrcoef, precision_score)
from sklearn.metrics import confusion_matrix
from functions import carom
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import os

#%%
# data setup
df_carom = pd.read_csv("../Supplementary Data/caromDataset.csv")

feature_names = df_carom.columns[3:16]   # index version
print(feature_names)

X = df_carom[feature_names]
y = df_carom["Target"]

# transform y classes from [-1,0,1] to [0,1,2]
le = preprocessing.LabelEncoder()
le.fit(y)
y = pd.Series(le.transform(y))
#%%
# prepare various models to test with cross-validation
models = []
models.append(('LR', LogisticRegression(max_iter=5000)))
models.append(('XGB', xgboost.XGBClassifier(n_estimators=100,max_depth=10,
                                            use_label_encoder=False,
                                            eval_metric="mlogloss")))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier(max_depth=8)))
models.append(('AB', AdaBoostClassifier(n_estimators=100)))
models.append(('RF', RandomForestClassifier(n_estimators=100,max_depth=10)))
models.append(('SVM', SVC()))

results = []
names = []
seed = 123
scoring = 'f1_macro'

for name, model in models:
	kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

#%%
## put results in dataframe and plot

df = pd.DataFrame(results).T
df.columns = names
print(df)
loop_stats = df.describe()

CIs = st.t.interval(alpha=0.95, df=len(df)-1,
          loc=np.mean(df), scale=st.sem(df))

# make folder for figures
path = "../figures/algorithm_comparison/"
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

pltFont = {'fontname':'Arial'}

# plot CV scores
plt.rcParams.update(plt.rcParamsDefault) 
plt.rcParams['xtick.major.pad']='10'
fig, ax = plt.subplots(figsize=(8,6))
bg=ax.bar(names, np.mean(df),
       yerr=loop_stats.loc['std',:],
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10,
       width=0.8)
ax.set_ylim([0, 1.0])
plt.yticks(**pltFont)
ax.set_xticklabels(names,**pltFont)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('Algorithm Comparison', fontsize=24,**pltFont)
ax.yaxis.grid(True)
plt.ylabel("F1 Score", fontsize=20, **pltFont)
bg[1].set_facecolor('r')
plt.tight_layout()
plt.savefig(path+'algorithmComparison_barGraph.png',
            bbox_inches='tight', dpi=600)
plt.show()
#%%

## further test AdaBoost and Random Forest by tuning hyperparameters

# set up AdaBoost parameters/model
AB_params = {'learning_rate' :  [0.001, 0.01, 0.1, 0.25, 0.5, 1.0]}
classifier_AB = AdaBoostClassifier(n_estimators=150,
                                random_state=123) #multi:softmax
random_search_AB = GridSearchCV(classifier_AB, param_grid=AB_params,
                             scoring='f1_macro',  # 100
                             n_jobs=-1, cv=5, verbose=3) 

# set up Random Forest parameters/model
max_features = ['auto', 'sqrt']
max_depth = [4,6,8,10]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
RF_params = {'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': min_samples_leaf}
classifier_RF = RandomForestClassifier(n_estimators=150,
                                       random_state=123)
random_search_RF = RandomizedSearchCV(classifier_RF, param_distributions=RF_params,
                             scoring='f1_macro',  # 100
                             n_jobs=-1, cv=5, verbose=3) 


models = []
models.append(('AdaBoost', random_search_AB))
models.append(('Random Forest', random_search_RF))

class_names = ["Phos","Unreg","Acetyl"]

avg = "macro"
        
cv = StratifiedKFold(n_splits=5,
                     shuffle=True, 
                     random_state=1)
num_class=3

# loop through RF and AdaBoost models
for name, model in models:
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
    cmCV = np.zeros((num_class, num_class))
    paramDict = {}
    
    count = 0
    
    for train_index, test_index in cv.split(X, y):
        X_trainCV, X_testCV = X.iloc[train_index], X.iloc[test_index]
        y_trainCV, y_testCV = y.iloc[train_index], y.iloc[test_index]
            
    
        model.fit(X_trainCV, y_trainCV)
        
        best_Mdl = model.best_estimator_
        print("Cross-val Fold {}, Model Params: {}".format(count, best_Mdl))
        paramDict[count] = best_Mdl.get_params
        y_predCV = best_Mdl.predict(X_testCV)
        
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
            
        acc_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        mcc_list.append(mcc)
        r_list.append(r)
    
        count = count+1
             
    # print final CM
    print("final CV confusion matrix: \n",cmCV)

    ### plot confusion matrix results 
    carom.make_confusion_matrix(y_test, y_pred, figsize=(8,6), categories=class_names,
                          xyplotlabels=True, cbar=False, sum_stats=False)
    plt.ylabel("Experimental Labels", fontsize=24)
    plt.xlabel("Predicted Labels", fontsize=24)
    plt.tight_layout()
    plt.savefig(path+"{}_crossval_confusionMat.png".format(name),
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
    
    CIs = st.t.interval(alpha=0.95, df=len(df_loop_scores)-1,
              loc=np.mean(df_loop_scores), scale=st.sem(df_loop_scores))
    
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
    ax.set_title('{} Classifer'.format(name), fontsize=24, **pltFont)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(path+'{}_crossVal_barGraph.png'.format(name),
                bbox_inches='tight', dpi=600)
    plt.show()

    