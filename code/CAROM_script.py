#!/usr/bin/env python
# coding: utf-8

"""
# 'CAROM_script.py' contains the main code for producing the models and figures
# described in the CAROM-ML paper. The goal of the CAROM-ML project is to use
# certain gene, reaction and protein properties to predict the type of 
# post-translational modification applied to these targets of metabolic
# regulation. 
#
# The script is broken into the following sections:
# Part 1: Data import and setup
# Part 2: Training the primary CAROM-ML model
# Part 3: Explaining the CAROM-ML model using the SHAP package
# Part 4: Analyzing the predictions on select genes of interest
# Part 5: Creating the SHAP force plots presented in the CAROM-ML paper
# Part 6: Validating the CAROM-ML model on the cell cycle G1/S/G2 datasets
# Part 7: Decision tree models for further interpretation/visualization of
#         the machine-learning problem.
#
# Note that ustom functions used through the script are stored in the "carom"
# library. The function parameters and returns are listed in this script, but
# can also be accessed use the help function (e.g help(carom.xgb_func)).

@author: Kirk Smith
"""

import pandas as pd
import numpy as np
import os
from functions import carom
import shap
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter
from sklearn import preprocessing
import xlsxwriter

    
### Part 1: Data import and setup ###

# main CAROM dataset (contains E.coli, yeast, HeLa and G0 training data)
df_carom = pd.read_csv("../Supplementary data/caromDataset.csv")

# cell cycle datasets (which use z-score for identifying d)
df_g0Z = pd.read_excel("../Supplementary data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_G0",engine='openpyxl')
df_g1Z = pd.read_excel("../Supplementary data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_G1",engine='openpyxl')
df_sZ = pd.read_excel("../Supplementary data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_S",engine='openpyxl')
df_g2Z = pd.read_excel("../Supplementary data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_G2",engine='openpyxl')
#%%

# view variables and data types
print(df_carom.dtypes)

# define feature names
feature_names = df_carom.columns[3:16]   # index version
print(feature_names)

# define class names
class_names = ["Phos","Unreg","Acetyl"]

# view number of gene-rxn pairs by class
print("Number of gene-rxn pairs per class :\n", df_carom['Target'].value_counts())

# view number of gene-rxn pairs by organism types
print("Number of gene-rxn pairs per organism-type:\n", df_carom['Organism'].value_counts())
#%%

### Part 2: Train XGBoost model ###

# Function Name: carom.xgb_func

# Function Inputs:
# 1. X:            predictor features matrix 
# 2. y:            target variable
# 3. num_iter:     number of cross-validation folds
# 4. condition:    string used to identify dataset condition (e.g "e.coli").
#                  Used to name/save plots.
# 5. class_names:  names of target classes   
# 6. depth:        Determines the range of values for tuning the max_depth 
#                  hyperparameter. Options are "deep" (default) or "shallow"
# 7. imbalance     Determines the method for addressing class imbalance:
#                  a.) List of two floats (for multi-class) or single float
#                      (for binary): Model will use SMOTE oversampling for the
#                      Phos/Acetyl classes. This assumes that Unregulated is
#                      largest class. Ex: [0.5, 0.75] --> Phos class will be 
#                      over-sampled to 50% of size of Unreg, Acetyl to 75% of
#                      Unreg.
#                  b.) "adasyn": uses Adasyn over-sampling method. Phos and                      
#                      Acetyl classes over-sampled to 75% of Unreg class. 
#                  c.) "undersample": undersamples larger classes to size of 
#                      smallest class. 
#                  d.) "none" (default): class balances are not adjusted
#                  f.) "weighted": inverse proportion of classes are used to
#                      assign class weights for "sample_weights" argument in 
#                      hyperparameter tuning
#       
# Function Outputs:
# 1. XGBoost model
# 2. Dataframe w/ XGBoost cross-val scores

# run carom.xgb_func
[caromMdl, caromScores]= carom.xgb_func(
   X=df_carom[feature_names],
   y=df_carom['Target'],
   num_iter=5,
   condition='MainCaromMdl', 
   class_names=class_names,
   depth="deep",
   imbalance="none")

# save model
filename1 = '../models/carom_XGBmodel.sav'
pickle.dump(caromMdl, open(filename1, 'wb'))

# to load model:
# caromMdl = pickle.load(open('../models/carom_XGBmodel.sav', 'rb'))

#%%
### Part 3: SHAP analysis ###

# Function Name: carom.shap_func

# Function Inputs:
# 1. xgbModel:    XGBoost model
# 2. X:           the independent data that you want explained 
# 3. condition:   string used to identify dataset for plots/files
# 4. class_names: stringg array of target class names 
        
# Function Outputs: 
# 1. explainer:   SHAP explainer object
# 2. shap_values: matrix of SHAP values
    
[carom_explainer, carom_shapValues] = carom.shap_func(
    xgbModel = caromMdl,
    X = df_carom[feature_names],
    condition = "CAROM",
    class_names = ["Phosphorylation","Unregulated","Acetylation"]) 

# save SHAP explainer
pickle.dump(carom_explainer, open('../models/carom_explainer.sav', 'wb'))

# to load explainer:
# carom_explainer = pickle.load(open('../models/carom_explainer.sav', 'rb'))

#%%
### Part 4: Analyzing predictions of select genes by gene name ###

# Function Name: carom.predict_genes

# Function Inputs:
# 1. X:            predictor features
# 2. y:            target variable
# 3. all_genes:    string array(s) of all gene IDs corresponding to the rows
#                  in the X and y datasets              
# 4. select_ganes: string array of gene names to be predicted
# 5. xgb_clf:      XGBoost classifier model
# 6. explainer:    SHAP explainer object
# 7. class_names:  string array of class names
# 8. condition:    string of dataset condition (for naming plots)   
        
# Function Outputs: 
# All plots are saved to the 'figures/predict_genes' folder.

X = df_carom[feature_names]
y = df_carom["Target"]

# create list of genes to predict
select_genes = ["YDL080C","PDHA1"] #"TKT","PLCB2","GCLM","CAD"

carom.predict_genes(X = X, y = y,
              all_genes = df_carom[['genes','reaction']],
              select_genes = select_genes,
              condition = "CAROM_NoW",
              class_names = class_names,
              xgb_clf = caromMdl,
              explainer = carom_explainer)

#%%
### Part 5: SHAP Force Plots ###

# This section is used to create the force plots in the CAROM-ML paper
X = df_carom[feature_names]
y = df_carom["Target"]

# transform target classes to [0, 1, 2] so they match model
y = y.reset_index(drop=True)
le = preprocessing.LabelEncoder()
le.fit(y)
y = pd.Series(le.transform(y))

# adjust test_size to get ~50 observations in "test" set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0039,
                                           stratify=y,
                                           random_state=999)
# get gene and reaction names for X_test
testInd = X_test.index
testGenes = df_carom.genes[testInd]
testRxns = df_carom.reaction[testInd]

# get model predictions for X_test
logodds = caromMdl.predict(X_test, output_margin=True)
ypred = caromMdl.predict(X_test)

# get SHAP values for X_test
explainer = carom_explainer
expected_value = explainer.expected_value
shap_explainer = explainer(X_test)
shap_values = explainer.shap_values(X_test)

# shortened feature names
feats_short = ['geneKO','maxATP','growthAC','close','degree','between',
               'pageRank','reverse','Vmin','Vmax','PFBA','Kcat',
               'MW']

# find correct predictions in test set for each class
phos_pairs = np.where((ypred==0) & (y_test==ypred))
print(phos_pairs)
acetyl_pairs = np.where((ypred==2) & (y_test==ypred))
print(acetyl_pairs)

# print log odds for correct predictions ()
print(logodds[phos_pairs],'\n')
print(logodds[acetyl_pairs])

# pick one observation each for Phos and Acetyl. This is a trial and 
# error process, looking for a good representative force plot
phos_index = 4
print(X_test.index[phos_index])
acetyl_index = 10
print(X_test.index[acetyl_index])

# get gene and reaction names
phosGene = testGenes.iloc[phos_index]
phosRxn = testRxns.iloc[phos_index]
print("Phos gene-rxn: {}-{}".format(phosGene, phosRxn))
acetylGene = testGenes.iloc[acetyl_index]
acetylRxn = testRxns.iloc[acetyl_index]
print("Acetyl gene-rxn: {}-{}".format(acetylGene, acetylRxn))

num_class = len(class_names)

# plot and save force plots
path = '../figures/SHAP/force_plots'
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

for which_class in range(0,num_class):
    # Single force plot for Phos observation (will make a plot for each class,
    # but we are mainly interested in plot for Phos class)
    p1 = shap.force_plot(base_value = expected_value[which_class],
                    shap_values = shap_values[which_class][phos_index],
                    plot_cmap=["#e80a89","#66AA00"],
                    features = X_test.iloc[phos_index],
                    feature_names = feats_short, show=True)
    shap.save_html('../figures/SHAP/force_plots/PhosForcePlot_{}_{}.html'.format(
        class_names[which_class], phosGene), p1)

    # Single force plot for Acetyl observation (will make a plot for each class,
    # but we are mainly interested in plot for Acetyl class)
    p2 = shap.force_plot(base_value = expected_value[which_class],
                    shap_values = shap_values[which_class][acetyl_index],
                    plot_cmap=["#e80a89","#66AA00"], #"PkYg"
                    features = X_test.iloc[acetyl_index],
                    feature_names = feats_short)
    shap.save_html('../figures/SHAP/force_plots/AcetylForcePlot_{}_{}.html'.format(
        class_names[which_class], acetylGene), p2)

    # Collective force plot for all 50 observations (one plot for each class)
    p3 = shap.force_plot(base_value=expected_value[which_class],
                    shap_values=shap_values[which_class],
                    features=X_test, 
                    feature_names=feats_short,
                    plot_cmap=["#e80a89","#66AA00"])
    shap.save_html('../figures/SHAP/force_plots/CollectiveForcePlot_{}GeneRxns_{}.html'.format(
        len(X_test), class_names[which_class]),p3)
    

#%%

### Part 6: XGBoost model validation using cell-cycle G1/S/G2 datasets ###

# Function Name: carom.mdl_predict

# Function Inputs:
# 1. mdl:             Classifier model
# 2. X:               Features matrix used for predictions
# 3. y:               True values of target variable. If given, output will
#                     include figures related to classification performance.
#                     If NOT given, model will only return predictions.             
# 4. condition:       string of dataset condition (for naming plots/files)
# 5. class_names:     Target variable class names
# 6. gene_reactions:  Dataframe with gene and reaction IDs corresponding to the 
#                     features matrix. Used to output names of predictions for each class.
# 7. confusion_mat:   Option to output classification confusion matrix
#                     (default=True).
# 8. bar:             Option to output bar graph with classification scores
#                     (default=False).
# 9. swarm:           Option to output matrix of swarmplots for all numerical
#                     features, grouped by true pos and false neg (default=False).
# 10. pairwise:       Option to output pairwise plot for top 5 most important
#                     features (default=False).
# 11. boxplot:        Option to output matrix of feature boxplots grouped by 
#                     features classification group (default=False).
# 12. explainer:      SHAP explainer object (default=None). If given, several
#                     plots are produced with the SHAP package. 
        
# Function Outputs: 
# If 'y' is provided...
# 1. scores:          model's classification scores
# 2. acetylGenesPred: list of genes/reactions predicted to be Acetyl
# 3. phosGenesPred:   list of genes/reactions predicted to be Phos
# 4. ypred:           array of model class predictions as integers 

# If 'y' is NOT provided...
# 1. acetylGenesPred: list of genes/reactions predicted to be Acetyl
# 2. phosGenesPred:   list of genes/reactions predicted to be Phos
# 3. ypred:           array of model class predictions as integers 


df_cellCycle = pd.concat([df_g1Z, df_sZ, df_g2Z])

X = df_cellCycle[feature_names]
y = df_cellCycle["Target"]
print(y.value_counts())

[scores, LeePredictAcetyl, LeePredictPhos, LeeYpred] = carom.mdl_predict(
                          mdl=caromMdl,
                          X=X, y=y,
                          condition="CellCycle", class_names=class_names,
                          gene_reactions=df_cellCycle[['genes','reaction','Phase']],
                          confusion_mat=True, bar=False, swarm=False,
                          pairwise=False, boxplot=False,
                          explainer=None)

# write unique genes and reactions to file (Phos only)
uniqueLeePredPhosGenes = LeePredictPhos.genes.unique()
uniqueLeePredPhosRxns = LeePredictPhos.reaction.unique()

array = [uniqueLeePredPhosGenes,
         uniqueLeePredPhosRxns]

workbook = xlsxwriter.Workbook('../results/CellCyclePhosPred_Unique.xlsx')
worksheet = workbook.add_worksheet()

row = 0
for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()
#%%
# check model predictions for genes that change between G0 vs G1/S/G2
for df_test in [df_g1Z, df_sZ, df_g2Z]:
    X = df_test.loc[:,feature_names]
    y = df_test.loc[:,"Target"]
    
    [score_test, predictAcetyl_test, predictPhos_test, ypred_test] = carom.mdl_predict(mdl=caromMdl,
                                  X=X, y=y,
                                  condition="Test", class_names=class_names,
                                  gene_reactions=df_test[['genes','reaction','Phase']], 
                                  confusion_mat=False, bar=False, swarm=False,
                                  pairwise=False, explainer=None)
    
    ix1 = (df_g0Z.Target.values!=df_test.Target.values)
    
    df_changed = pd.DataFrame({'Genes' : df_g0Z.genes[ix1],
                               'Control' : df_g0Z.Target.values[ix1],
                               'Test_True' : df_test.Target.values[ix1],
                               'Test_Pred' : ypred_test[ix1]-1
                               })
    print(df_changed)
    print(Counter(df_changed.Test_Pred))
#%%

### Part 7: Create decision tree models for visualization ###

# Function Name: carom.decisionTree

# Function Inputs:
# 1. X:            dataframe of features matrix
# 2. y:            target variable array
# 3. class_names:  string array of class names
# 4. weighting:    options for handling class imbalance
#                  a.) "none": no class weights or sampling applied
#                  b.) "balanced" (default): inverse class proportions
#                      used to assign class weights
#                  c.) "smote": SMOTE over-sampling applied 
#                  d.) "tuned": weights are tuned via cross-validation
# 5. weights       dictionary of weights to use with the "tuned" weighting
#                  option   
# 6. pruneLevel    integer, designating the minimum number of observations
#                  in a leaf for it to be pruned  
# 7. condition:    string of dataset condition (for naming plots/files)
                 
        
# Function Outputs: 
# Decision tree plots are saved to the 'figures/decision_trees' folder.
# The plots are saved as PDF files, which also include a confusion matrix
# for the model's resampling classification results.


# define weights for tuned option
tuning_weights = [{-1:25,0:1,1:10},{-1:25,0:1,1:20},
           {-1:30,0:1,1:15},{-1:30,0:1,1:20},
           {-1:35,0:1,1:15}, {-1:35,0:1,1:20},
           {-1:40,0:1,1:15}, {-1:40,0:1,1:20}]

X = df_carom[feature_names]
y = df_carom["Target"]

carom.decisionTree(X, y, class_names=class_names,
              weighting="balanced",
              condition = "AllFeats_Balanced")

carom.decisionTree(X, y, class_names=class_names,
              weighting="none",
              condition = "AllFeats_NoWeights")

carom.decisionTree(X, y, class_names=class_names,
              weighting="smote",
              condition = "AllFeats_SMOTE")

carom.decisionTree(X, y, class_names=class_names,
              weighting="tuned",
              weights=tuning_weights,
              condition = "AllFeats_TunedW")


