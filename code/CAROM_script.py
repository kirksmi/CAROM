#!/usr/bin/env python
# coding: utf-8

# In[69]:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import seaborn as sns
# from CAROM_functions import (multi_heatmap, predict_genes,
#                            xgb_func, shap_func,
#                            make_confusion_matrix,
#                            corr_heatmap, decisionTree,
#                            mdl_predict)
from functions import CAROM_functions2

import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
# from imblearn.under_sampling import NearMiss 
# from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import preprocessing
import xlsxwriter

# In[90]:


### Import datasets ###

# primary dataset
df_carom = pd.read_csv("../data/caromDataset.csv")

df_g0Z = pd.read_excel("../data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_G0")
df_g1Z = pd.read_excel("../data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_G1")
df_sZ = pd.read_excel("../data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_S")
df_g2Z = pd.read_excel("../data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_G2")
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
##### Machine-learning analysis #####

## Part 1: train XGBoost model

# Function Name: xgb_func

# Function Inputs:
# 1. X:            predictor features
# 2. y:            target variable
# 3. num_iter:     number of cross-validation folds
# 4. condition:    string used to identify dataset condition (e.g "e.coli").
#                  Used to name/save plots.
# 5. class_names:  names of target classes        
        
# Function Outputs:
# 1. XGBoost model
# 2. Dataframe w/ XGBoost cross-val scores

#%%

# run w/ no weights: Norm2010
[caromMdl, caromScores]= CAROM_functions2.xgb_func(
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

 #%%
# load model
caromMdl = pickle.load(open('../models/carom_XGBmodel.sav', 'rb'))
caromMdl_shallow = pickle.load(open('../models/carom_XGBmodel_shallow.sav', 'rb'))
caromMdl_NoG0 = pickle.load(open('../models/carom_XGBmodel_NoG0.sav', 'rb'))

#%%
## Part 2: SHAP analysis

# Function Name: shap_func

# Function Inputs:
# 1. xgbModel:    XGBoost model
# 2. X:           the independent data that you want explained by 
#                 the SHAP values.
# 3. condition:   string used to identify dataset condition
# 4. class_names: names of target classes 
        
# Function Outputs: 
# 1. explainer:   SHAP explainer object
# 2. shap_values: matrix of SHAP values
    
[carom_explainer, carom_shapValues] = CAROM_functions2.shap_func(
    xgbModel = caromMdl,
    X = df_carom[feature_names],
    condition = "CAROM",
    class_names = ["Phosphorylation","Unregulated","Acetylation"]) 

pickle.dump(carom_explainer, open('../models/carom_explainer.sav', 'wb'))
carom_explainer = pickle.load(open('../models/carom_explainer.sav', 'rb'))



#%%
## Part 3: Predicting select genes

# Function Name: predict_genes

# Function Inputs:
# 1. X:            predictor features
# 2. y:            target variable
# 3. all_genes:    string array of all genes corresponding to the rows
#                  in the X and y datasets              
# 4. select_ganes: string array of genes to be predicted
# 5. xgb_clf:      XGBoost classifier model
# 6. explainer:    SHAP explainer object
# 7. class_names:  string array of class names
# 8. condition:    string of dataset condition (for naming plots)   
        
# Function Outputs: 
# All plots are saved to the 'figures' folder.

X = df_carom[feature_names]
y = df_carom["Target"]

# create list of genes to predict
select_genes = ["YDL080C"] #"TKT","PDHA1","PLCB2","GCLM","CAD"

CAROM_functions2.predict_genes(X = X, y = y,
              all_genes = df_carom[['genes','reaction']],
              select_genes = select_genes,
              condition = "CAROM_NoW", class_names = class_names,
              xgb_clf = caromMdl,
              explainer = carom_explainer)

#%%

## Part 4: SHAP Force Plots 

# The following section was used to create the force plots in 
# the CAROM-ML paper

# Only want ~50 observations in collective force plot --> make
# stratified split on dataset

X = df_carom[feature_names]
y = df_carom["Target"]

y = y.reset_index(drop=True)
le = preprocessing.LabelEncoder()
le.fit(y)
y = pd.Series(le.transform(y))

# adjust test_size to get ~50 observations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0039,
                                           stratify=y,
                                           random_state=999)

# get gene names for X_test
testInd = X_test.index
testGenes = df_carom.genes[testInd]
testRxns = df_carom.reaction[testInd]

features = X_test
logodds = caromMdl.predict(features, output_margin=True)
ypred = caromMdl.predict(features)
ytrue = y_test

explainer = carom_explainer
expected_value = explainer.expected_value
shap_explainer = explainer(features)
shap_values = explainer.shap_values(features)

feats_short = ['geneKO','maxATP','growthAC','close','degree','between',
               'pageRank','reverse','rawVmin','rawVmax','PFBA','Kcat',
               'MW']

phos_pairs = np.where((ypred==0) & (ytrue==ypred))
print(phos_pairs)
acetyl_pairs = np.where((ypred==2) & (ytrue==ypred))
print(acetyl_pairs)

print(logodds[phos_pairs],'\n')
print(logodds[acetyl_pairs])


phos_index = 4
print(features.index[phos_index])
acetyl_index = 10
print(features.index[acetyl_index])

class_names = class_names
num_class = len(class_names)

phosGene = testGenes.iloc[phos_index]
phosRxn = testRxns.iloc[phos_index]
print("Phos gene-rxn: {}-{}".format(phosGene, phosRxn))
acetylGene = testGenes.iloc[acetyl_index]
acetylRxn = testRxns.iloc[acetyl_index]
print("Acetyl gene-rxn: {}-{}".format(acetylGene, acetylRxn))

# plot and save force plots
path = '../figures/SHAP/force_plots'
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

for which_class in range(0,num_class):
    ## Phos
    p1 = shap.force_plot(base_value = expected_value[which_class],
                    shap_values = shap_values[which_class][phos_index],
                    plot_cmap=["#e80a89","#66AA00"],
                    features = features.iloc[phos_index],
                    feature_names = feats_short, show=True)
    shap.save_html('../figures/SHAP/force_plots/PhosForcePlot_{}_{}.html'.format(
        class_names[which_class], phosGene), p1)

    # Acetyl
    p2 = shap.force_plot(base_value = expected_value[which_class],
                    shap_values = shap_values[which_class][acetyl_index],
                    plot_cmap=["#e80a89","#66AA00"], #"PkYg"
                    features = features.iloc[acetyl_index],
                    feature_names = feats_short)
    shap.save_html('../figures/SHAP/force_plots/AcetylForcePlot_{}_{}.html'.format(
        class_names[which_class], acetylGene), p2)

     # collective force plot
    p3 = shap.force_plot(base_value=expected_value[which_class],
                    shap_values=shap_values[which_class],
                    features=features, #X_testR.iloc[row_range,:]
                    feature_names=feats_short,
                    plot_cmap=["#e80a89","#66AA00"])
    shap.save_html('../figures/SHAP/force_plots/CollectiveForcePlot_{}GeneRxns_{}.html'.format(
        len(features), class_names[which_class]),p3)
    

#%%

## Part 5: Use XGBoost model to predict Lee G1/S/G2 conditions 

df_cellCycle = pd.concat([df_g1Z, df_sZ, df_g2Z])
df = df_cellCycle

X = df.loc[:,feature_names]
y = df.loc[:,"Target"]
print(y.value_counts())

[scores, LeePredictAcetyl, LeePredictPhos, LeeYpred] = CAROM_functions2.mdl_predict(
                          mdl=caromMdl, X=X, y=y,
                          condition="CellCycle", class_names=class_names,
                          gene_reactions=df[['genes','reaction','Phase']],
                          confusion_mat=True, bar=False, swarm=False,
                          pairwise=False, boxplot=False, explainer=None,
                          gscatter=False)

# write unique genes and reactions to file (Phos only)
uniqueLeePredPhosGenes = LeePredictAcetyl.genes.unique()
uniqueLeePredPhosRxns = LeePredictAcetyl.reaction.unique()

array = [uniqueLeePredPhosGenes,
         uniqueLeePredPhosRxns]

workbook = xlsxwriter.Workbook('../results/LeePhosPred_Unique.xlsx')
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
    
    [score_test, predictAcetyl_test, predictPhos_test, ypred_test] = CAROM_functions2.mdl_predict(mdl=caromMdl,
                                  X=X, y=y,
                                  condition="Test", class_names=class_names,
                                  gene_reactions=df_test[['genes','reaction','Phase']], 
                                  confusion_mat=False, bar=False, swarm=False,
                                  pairwise=False, explainer=None, gscatter=False)
    
    ix1 = (df_g0Z.Target.values!=df_test.Target.values)
    
    df_changed = pd.DataFrame({'Genes' : df_g0Z.genes[ix1],
                               'Control' : df_g0Z.Target.values[ix1],
                               'Test_True' : df_test.Target.values[ix1],
                               'Test_Pred' : ypred_test[ix1]-1
                               })
    print(df_changed)
    print(Counter(df_changed.Test_Pred))
#%%

## Part 6: Create decision trees for visualization

# weighting options: balanced (default), tuned, smote, none

# define weights for tuned option
balance = [{-1:25,0:1,1:10},{-1:25,0:1,1:20},
           {-1:30,0:1,1:15},{-1:30,0:1,1:20},
           {-1:35,0:1,1:15}, {-1:35,0:1,1:20},
           {-1:40,0:1,1:15}, {-1:40,0:1,1:20}]
X = df_carom[feature_names]
y = df_carom["Target"]


CAROM_functions2.decisionTree(X, y, class_names=class_names,
              weighting="balanced",
              condition = "AllFeats_Balanced")

CAROM_functions2.decisionTree(X, y, class_names=class_names,
              weighting="none",
              condition = "AllFeats_NoWeights")

CAROM_functions2.decisionTree(X, y, class_names=class_names,
              weighting="smote",
              condition = "AllFeats_SMOTE")

CAROM_functions2.decisionTree(X, y, class_names=class_names,
              weighting="tuned",
              condition = "AllFeats_TunedW")


