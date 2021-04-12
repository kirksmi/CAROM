#!/usr/bin/env python
# coding: utf-8

"""
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
2. y_train:            target variable
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
                 f.) "weighted": inverse proportion of classes are used to
                      assign class weights for "sample_weights" argument in 
                      hyperparameter tuning
       
Function Outputs:
---------
1. XGBoost model
2. Dataframe w/ XGBoost cross-val scores

@author: kirksmi
"""
#%%
import sys
import pandas as pd
from functions import carom

df_train = pd.read_csv(sys.argv[1])
df_test = pd.read_csv(sys.argv[2])
class_names = sys.argv[3]

condition = "CAROM_train"
# define feature names
feature_names = ['geneKO', 'maxATPafterKO', 'growthAcrossCond', 'closeness',
                 'degree', 'betweenness', 'pagerank', 'reversible', 'rawVmin',
                 'rawVmax', 'PFBAflux', 'kcat', 'MW']

X_train = df_train[feature_names].sample(100)
y_train = df_train["Target"][X_train.index]

[model, scores]= carom.xgb_func(X=X_train, y=y_train,
                                num_iter=3,
                                condition=condition,
                                class_names=class_names,
                                depth="shallow",
                                imbalance="none")
                                
    
    # [S_acetylGenes, S_phosGenes, S_ypred] = carom.mdl_predict(
    #                                                 mdl=caromMdl_NoG0,
    #                                                 X=df_sZ[feature_names],
    #                                                 condition="PredictS_NoG0Mdl",
    #                                                 class_names=class_names,
    #                                                 gene_reactions=df_sZ[['genes','reaction']])
    
    # # transform Target classes to 0, 1, 2
    # le = preprocessing.LabelEncoder()
    # le.fit(y_train)
    # y_train = pd.Series(le.transform(y_train))
    
    # # if class names not given, use class integers
    # if class_names is None:
    #     class_names=[]
    #     for cl in y_train.unique():
    #         class_names.append(np.array2string(cl))
    # num_class = len(np.unique(y_train))
    # print("Number of class: {}".format(num_class))
   
    # # hyperparameters to tune 
    # # (max_depth adjusted based on 'depth' argument)
    # if depth == "shallow":
    #     params={
    #     "learning_rate"    : [0.01, 0.05, 0.1, 0.3],
    #     "max_depth"        : range(4,6,1), #range(4,11,2),
    #     "min_child_weight" : [3, 5, 7],
    #     "subsample"        : [0.8, 0.9],
    #     "colsample_bytree" : [0.8, 0.9],
    #     }
    # else:
    #     params={
    #     "learning_rate"    : [0.01, 0.05, 0.1, 0.3],
    #     "max_depth"        : range(4,11,2),
    #     "min_child_weight" : [3, 5, 7],
    #     "subsample"        : [0.8, 0.9],
    #     "colsample_bytree" : [0.8, 0.9],
    #     }

    # ##### Train model using cross-val and hyperparameter tuning #####   

    # # Define classifiers and hyperparameter search, based on binary vs multi-class problem
    # if num_class == 2:  # binary model
    #     print("TRAINING BINARY MODEL!")
    #     # define classifier and hyperparameter tuning
    #     classifier = xgboost.XGBClassifier(objective='binary:logistic',
    #                                        n_estimators=150,
    #                                        use_label_encoder=False,
    #                                        eval_metric='logloss',
    #                                        random_state=123)
    
    #     random_search = RandomizedSearchCV(classifier, param_distributions=params,
    #                                       n_iter=30, scoring='f1',  # 100
    #                                       n_jobs=-1, cv=5, verbose=3,
    #                                       random_state=123) 
    #     avg = "binary"
    
    # elif num_class > 2: # multi-class model
    #     print("TRAINING MULTI-CLASS MODEL!")
    #     classifier = xgboost.XGBClassifier(objective='multi:softmax',
    #                                        n_estimators=150,
    #                                        use_label_encoder=False,
    #                                        num_class=num_class,
    #                                        eval_metric='mlogloss',
    #                                        random_state=123) #multi:softmax

    #     random_search = RandomizedSearchCV(classifier, param_distributions=params,
    #                                       n_iter=30, scoring='f1_macro',  # 100
    #                                       n_jobs=-1, cv=5, verbose=3,
    #                                       random_state=123) 
    #     avg = "macro"
        
    # # Stratified cross-val split
    # cv = StratifiedKFold(n_splits=5,
    #                      shuffle=True, 
    #                      random_state=123)
    
    # # create empty lists to store CV scores, confusion mat, etc.
    # acc_list = []
    # recall_list = []
    # precision_list = []
    # f1_list = []
    # mcc_list = []
    # r_list = []
    
    # y_testCV_all = []
    # y_predCV_all = []
    # cmCV = np.zeros((num_class, num_class))
    
    # paramDict = {}

    # count = 0
    
    # # loop through cross-val folds
    # for train_index, test_index in cv.split(X_train, y_train):
    #     X_trainCV, X_testCV = X_train.iloc[train_index], X_train.iloc[test_index]
    #     y_trainCV, y_testCV = y_train.iloc[train_index], y_train.iloc[test_index]
            
    #     # tune hyperparameters
    #     random_search.fit(X_trainCV, y_trainCV)
    #     # get best estimator from random search
    #     randomSearch_mdl = random_search.best_estimator_
            
    #     # tune gamma and get new best estimator
    #     params_gamma = {'gamma':[0, 0.1, 0.3, 0.5]}
    #     gamma_search = GridSearchCV(estimator = randomSearch_mdl, 
    #                         param_grid = params_gamma, scoring='f1_macro',
    #                         n_jobs=-1 , cv=3)
    #     gamma_search.fit(X_trainCV, y_trainCV)
    #     best_Mdl = gamma_search.best_estimator_
        
    #     # print and store best params for current fold
    #     print("Cross-val Fold {}, Model Params: {}".format(count, best_Mdl))
    #     paramDict[count] = best_Mdl.get_params
        
    #     # make model predictions on X_testCV and store results
    #     y_predCV = best_Mdl.predict(X_testCV)        
    #     y_testCV_all.extend(y_testCV)
    #     y_predCV_all.extend(y_predCV)
        
    #     cm = confusion_matrix(y_testCV, y_predCV)
    #     print("current cm: \n",cm)
    #     # update overal confusion mat
    #     cmCV = cmCV+cm
    #     print("Combined cm: \n", cmCV)
        
    #     # calculate classification scores and store
    #     accuracy = accuracy_score(y_testCV, y_predCV)
    #     f1 = f1_score(y_testCV, y_predCV, average=avg)
    #     recall = recall_score(y_testCV, y_predCV, average=avg)
    #     precision = precision_score(y_testCV, y_predCV, average=avg)
    #     mcc = matthews_corrcoef(y_testCV, y_predCV)
    #     r = np.corrcoef(y_testCV, y_predCV)[0, 1]
            
    #     acc_list.append(accuracy)
    #     recall_list.append(recall)
    #     precision_list.append(precision)
    #     f1_list.append(f1)
    #     mcc_list.append(mcc)
    #     r_list.append(r)
    
    #     count = count+1
             
    # # print final confusion mat
    # print("final CV confusion matrix: \n",cmCV)
   
    # ### plot confusion matrix results 
    # path = '../figures/crossval/'
    # try:
    #     os.makedirs(path)
    # except OSError:
    #     print("Directory already created")
        
    # carom.make_confusion_matrix(y_testCV_all, y_predCV_all, figsize=(8,6), categories=class_names,
    #                       xyplotlabels=True, cbar=False, sum_stats=False)
    # plt.ylabel("Experimental Labels", fontsize=24)
    # plt.xlabel("Predicted Labels", fontsize=24)
    # plt.tight_layout()
    # plt.savefig("../figures/crossval/{}_XGBcrossval_confusionMat.png".format(condition),
    #                dpi=600)
    # plt.show()  
    
    # # get average scores
    # Accuracy = np.mean(acc_list)
    # F1 = np.mean(f1_list)
    # Precision = np.mean(precision_list)
    # Recall = np.mean(recall_list)
    # MCC = np.mean(mcc_list)
    # Corr = np.mean(r_list)
    
    # scores = [Accuracy, Recall, Precision, F1, MCC, Corr] #AUC,
    
    # # get stats for CV scores
    # loop_scores = {'Accuracy':acc_list,
    #                'Recall':recall_list,
    #                'Precision':precision_list,
    #                'F1':f1_list,
    #                'MCC':mcc_list,
    #                'R':r_list}
                    
    # df_loop_scores = pd.DataFrame(loop_scores)
    # print("Model score statistics: ")
    # loop_stats = df_loop_scores.describe()
    # print(loop_stats)
    
    # # plot CV scores
    # plt.rcParams.update(plt.rcParamsDefault) 
    # plt.rcParams['xtick.major.pad']='10'
    # fig, ax = plt.subplots(figsize=(8,6))
    # ax.bar(df_loop_scores.columns, scores,
    #        yerr=loop_stats.loc['std',:],
    #        align='center',
    #        alpha=0.5,
    #        ecolor='black',
    #        capsize=10,
    #        width=0.8)
    # ax.set_ylim([0, 1.0])
    # plt.yticks(**pltFont)
    # ax.set_xticks(df_loop_scores.columns)
    # ax.set_xticklabels(df_loop_scores.columns,**pltFont,
    #                    rotation=45, ha="right", rotation_mode="anchor")
    # ax.tick_params(axis='both', which='major', labelsize=24)
    # ax.yaxis.grid(True)
    # plt.tight_layout()
    # plt.savefig('../figures/crossval/{}_XGB_crossVal_barGraph.png'.format(condition),
    #             bbox_inches='tight', dpi=600)
    # plt.show()

    # # create dataframe with mean scores
    # data = {'Metric':['Acc', 'Recall', 'Precision','F1', 'MCC', 'PearsonsR'], 
    #   'Scores':[Accuracy, Recall, Precision, F1, MCC, Corr]} 
    # df_scores = pd.DataFrame(data)
    # df_scores = df_scores.set_index('Metric')

    # ### train model on entire training dataset using params from best CV model     
    # maxpos = mcc_list.index(max(mcc_list))
    # final_params = paramDict[maxpos]
    # print("CV MCCs: {}".format(mcc_list))
    # print("Best parameters: ", final_params)
    # final_Mdl = classifier
    # final_Mdl.get_params = final_params
    # final_Mdl.fit(X_train, y_train)
        

    # ### Feature importances
    # importances = final_Mdl.feature_importances_
    #  # Sort in descending order
    # indices = np.argsort(importances)[::-1]
    #  # Rearrange feature names so they match the sorted feature importances
    # names = [feature_names[i] for i in indices]   # for sfs
    
    #  # Create plot
    # plt.figure()
    # plt.bar(range(X_train.shape[1]), importances[indices]) 
    # plt.title("XGBoost Feature Importance")
    # plt.xticks(range(X_train.shape[1]), names,
    #             fontsize=18, rotation=45, horizontalalignment="right")
    # plt.yticks(fontsize=20)
    # plt.bar(range(X_train.shape[1]), importances[indices])  
    # plt.savefig("../figures/crossval/{}_XGB_featureImps.png".format(condition),
    #               bbox_inches='tight', dpi=600)
    # plt.show()
    
    # return final_Mdl, loop_stats

