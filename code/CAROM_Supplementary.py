#!/usr/bin/env python
# coding: utf-8

# In[69]:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from functions import CAROM_functions2 
import copy
import shap
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter
# import xlsxwriter

# In[90]:


### Import datasets ###

# primary dataset
df_carom = pd.read_csv("../data/caromDataset_NormG0.csv")
# create dataframes for each organism type
df_ecoli = df_carom.loc[df_carom.Organism==1]
df_yeast = df_carom.loc[df_carom.Organism==2]
df_human = df_carom.loc[df_carom.Organism==3]
df_noLee = pd.read_csv("../data/caromDataset_NormNoLee_2010.csv")
df_noBackground =pd.read_csv("../data/caromDataset_NoBackground.csv")

df_LeeNormZ = pd.read_csv("../data/Lee_NormZ.csv")
df_LeeG0 = pd.read_csv("../data/LeeG0_Norm.csv")

df_g1Z = df_LeeNormZ.loc[df_LeeNormZ.Phase=="G1"]
df_sZ = df_LeeNormZ.loc[df_LeeNormZ.Phase=="S"]
df_g2Z = df_LeeNormZ.loc[df_LeeNormZ.Phase=="G2"]


#%%
# view variables and data types
print(df_carom.dtypes)

# define feature names
feature_names = df_carom.columns[3:16]   # index version
print("Predictor features: ", feature_names)

feats_list = list(feature_names)   # list version
featsPlusOrg = feats_list + ["Organism"]

# define class names
class_names = ["Phos","Unreg","Acetyl"]

#%%
# Feature correlation

# define path name for saving trees
path = "../figures/correlation_heatmaps"

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

# plot correlation heatmaps
CAROM_functions2.corr_heatmap(df_ecoli[feature_names], "ecoli")
CAROM_functions2.corr_heatmap(df_yeast[feature_names], "yeast")
CAROM_functions2.corr_heatmap(df_human[feature_names], "human")
CAROM_functions2.corr_heatmap(df_carom, "allOrganisms")

#%%
X = df_carom[feature_names]
y = df_carom["Target"]
# view number of gene-rxn pairs by class
print("Number of regulated gene-rxn pairs:\n", df_carom['Target'].value_counts())
# view number of gene-rxn pairs by organism types
print("Number of each cell type:\n", df_carom['Organism'].value_counts())

#%%
##### Machine-learning analysis #####

## Part 1: train XGBoost model

# Function Name: xgb_func

# Function Inputs:
# 1. X:            predictor features
# 2. y:            target variable
# 3. num_iter:     number of iterations to for cross-validation
# 4. condition:    string used to identify dataset condition (e.g "e.coli").
#                  Used to name/save plots.
# 5. class_names:  names of target classes        
        
# Function Outputs:
# 1. XGBoost model
# 2. Dataframe w/ XGBoost cross-val scores

#%%
    
## run models for each organism type
# E. coli
[ecoli_mdl, ecoli_scores]= CAROM_functions2.xgb_func(
   X=df_ecoli[feature_names],
   y=df_ecoli['Target'],
   num_iter=5,
   condition='Ecoli_NoWeights', 
   class_names=class_names,
   imbalance="none")
# save model
filename1 = '../models/ecoli_XGBmodel.sav'
pickle.dump(ecoli_mdl, open(filename1, 'wb'))

# yeast
[yeast_mdl, yeast_scores]= CAROM_functions2.xgb_func(
   X=df_yeast[feature_names],
   y=df_yeast['Target'],
   num_iter=5,
   condition='Yeast_NoWeights', 
   class_names=class_names,
   imbalance="none")
# save model
filename1 = '../models/yeast_XGBmodel.sav'
pickle.dump(yeast_mdl, open(filename1, 'wb'))


[human_mdl, human_scores]= CAROM_functions2.xgb_func(
   X=df_human[feature_names],
   y=df_human['Target'],
   num_iter=5,
   condition='Human_NoWeights', 
   class_names=class_names,
   imbalance="none")
# save model
filename1 = '../models/human_XGBmodel.sav'
pickle.dump(human_mdl, open(filename1, 'wb'))

# run model with organism-type included as feature
[caromOrg_mdl, caromOrg_scores]= CAROM_functions2.xgb_func(
   X=df_carom[featsPlusOrg],
   y=df_carom['Target'],
   num_iter=5,
   condition='caromOrg_NoW', 
   class_names=class_names,
   imbalance="none")
# save model
filename1 = '../models/caromOrg_XGBmodel.sav'
pickle.dump(caromOrg_mdl, open(filename1, 'wb'))

# run model with no cell-cycle data included
[caromMdl_NoG0, caromScores_NoG0]= CAROM_functions2.xgb_func(
   X=df_noLee[feature_names],
   y=df_noLee['Target'],
   num_iter=5,
   condition='NoG0', 
   class_names=class_names,
   imbalance="none")
# save model
filename1 = '../models/carom_XGBmodel_NoG0.sav'
pickle.dump(caromMdl_NoG0, open(filename1, 'wb'))

# run model with no removal of background genes
[caromMdl_background, caromScores_background]= CAROM_functions2.xgb_func(
   X=df_noBackground[feature_names],
   y=df_noBackground['Target'],
   num_iter=5,
   condition='NoBackground', 
   class_names=class_names,
   imbalance="none")
# save model
filename1 = '../models/carom_XGBmodel_NoBackground.sav'
pickle.dump(caromMdl_background, open(filename1, 'wb'))
#%%

# train models with different cell-cycle phase datasets
# G1
df_caromG1 = pd.concat([df_noLee, df_g1Z])

[caromMdl_G1, caromScores_G1]= CAROM_functions2.xgb_func(
   X=df_caromG1[feature_names],
   y=df_caromG1['Target'],
   num_iter=5,
   condition='G1', 
   class_names=class_names,
   imbalance="none")
# save model
pickle.dump(caromMdl_G1, open('../models/carom_XGBmodel_G1.sav', 'wb'))

# S
df_caromS = pd.concat([df_noLee, df_sZ])

[caromMdl_S, caromScores_S]= CAROM_functions2.xgb_func(
   X=df_caromS[feature_names],
   y=df_caromS['Target'],
   num_iter=5,
   condition='S', 
   class_names=class_names,
   imbalance="none")
# save model
pickle.dump(caromMdl_S, open('../models/carom_XGBmodel_S.sav', 'wb'))

# G2
df_caromG2 = pd.concat([df_noLee, df_g2Z])

[caromMdl_G2, caromScores_G2]= CAROM_functions2.xgb_func(
   X=df_caromG2[feature_names],
   y=df_caromG2['Target'],
   num_iter=5,
   condition='G2', 
   class_names=class_names,
   imbalance="none")
# save model
pickle.dump(caromMdl_G2, open('../models/carom_XGBmodel_G2.sav', 'wb'))
 #%%
# load the models from disk
ecoli_mdl = pickle.load(open('../models/ecoli_XGBmodel.sav', 'rb'))
yeast_mdl = pickle.load(open('../models/yeast_XGBmodel.sav', 'rb'))
human_mdl = pickle.load(open('../models/human_XGBmodel.sav', 'rb'))
caromOrg_mdl = pickle.load(open('../models/caromOrg_XGBmodel.sav', 'rb'))
caromMdl_NoG0 = pickle.load(open('../models/carom_XGBmodel_NoG0.sav', 'rb'))
caromMdl = pickle.load(open('../models/carom_XGBmodel.sav', 'rb'))

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
    
[ecoli_explainer, ecoli_shapValues] = CAROM_functions2.shap_func(
    xgbModel = ecoli_mdl,
    X = df_ecoli[feature_names],
    condition = "Ecoli",
    class_names = ["Phosphorylation","Unregulated","Acetylation"]) 

[yeast_explainer, yeast_shapValues] = CAROM_functions2.shap_func(
    xgbModel = yeast_mdl,
    X = df_yeast[feature_names],
    condition = "Yeast",
    class_names = ["Phosphorylation","Unregulated","Acetylation"])

[human_explainer, human_shapValues] = CAROM_functions2.shap_func(
    xgbModel = human_mdl,
    X = df_human[feature_names],
    condition = "Human",
    class_names = ["Phosphorylation","Unregulated","Acetylation"]) 

[caromOrg_explainer, caromOrg_shapValues] = CAROM_functions2.shap_func(
    xgbModel = caromOrg_mdl,
    X = df_carom[featsPlusOrg],
    condition = "CAROMwithOrg",
    class_names = ["Phosphorylation","Unregulated","Acetylation"]) 


#%%
# make predictions for each cell cycle phase

[df_predictG0, G0_phosGenes, G0_acetylGenes] = CAROM_functions2.mdl_predict(
                                                    mdl=caromMdl_NoG0,
                                                    X=df_LeeG0[feature_names],
                                                    condition="LeeG0",
                                                    class_names=class_names,
                                                    gene_reactions=df_LeeG0[['genes','reaction']])
G0_acetylGenes["Phase"] = np.tile("G0",(len(G0_acetylGenes),1))

[df_predictG1, G1_phosGenes, G1_acetylGenes] = CAROM_functions2.mdl_predict(
                                                    mdl=caromMdl_NoG0,
                                                    X=df_g1Z[feature_names],
                                                    condition="LeeG1",
                                                    class_names=class_names,
                                                    gene_reactions=df_g1Z[['genes','reaction']])
G1_acetylGenes["Phase"] = np.tile("G1",(len(G1_acetylGenes),1))


[df_predictS, S_phosGenes, S_acetylGenes] = CAROM_functions2.mdl_predict(
                                                    mdl=caromMdl_NoG0,
                                                    X=df_sZ[feature_names],
                                                    condition="LeeS",
                                                    class_names=class_names,
                                                    gene_reactions=df_sZ[['genes','reaction']])
S_acetylGenes["Phase"] = np.tile("S",(len(S_acetylGenes),1))


[df_predictG2, G2_phosGenes, G2_acetylGenes] = CAROM_functions2.mdl_predict(
                                                    mdl=caromMdl_NoG0,
                                                    X=df_g2Z[feature_names],
                                                    condition="LeeG2",
                                                    class_names=class_names,
                                                    gene_reactions=df_g2Z[['genes','reaction']])
G2_acetylGenes["Phase"] = np.tile("G2",(len(G2_acetylGenes),1))

# combine predictions and write to file
LeeAcetylPred_AllPhases = pd.concat([G0_acetylGenes,G1_acetylGenes,
                                            S_acetylGenes, G2_acetylGenes])
LeeAcetylPred_AllPhases.to_csv('../results/LeeAcetylPred_AllPhases.csv',
                                index=False)

# get unique Acetyl genes and rxns
uniqueLeeAcetylGenes = LeeAcetylPred_AllPhases.genes.unique()
uniqueLeeAcetylRxns = LeeAcetylPred_AllPhases.reaction.unique()

array = [uniqueLeeAcetylGenes,
         uniqueLeeAcetylRxns]

workbook = xlsxwriter.Workbook('../results/LeeAcetylPred_Unique.xlsx')
worksheet = workbook.add_worksheet()

array = [uniqueLeeAcetylGenes,
         uniqueLeeAcetylRxns]

row = 0

for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()

# plot number of acetyl predictions per phase
acetylCounts = list(Counter(LeeAcetylPred_AllPhases.Phase).values())
fig,ax = plt.subplots()
sns.barplot(x=["G0","G1","S","G2"],y=acetylCounts)
plt.ylabel("# of Acetyl. Predictions",fontsize=16)
ax.tick_params(axis="both", labelsize=16)

#%%

# make prediction for the various cell-cycle models
df = pd.concat([df_LeeG0, df_sZ, df_g2Z])
[cm_g1, scores_g1, G1_PredictAcetyl, G1_PredictPhos] = CAROM_functions2.mdl_predict(
                          mdl=caromMdl_G1,
                          X=df.loc[:,feature_names],
                          y=df.loc[:,"Target"],
                          condition="G1_mdl", class_names=class_names,
                          gene_reactions=df[['genes','reaction','Phase']],
                          confusion_mat=True, bar=True, swarm=False,
                          pairwise=False, boxplot=False, explainer=None,
                          gscatter=False)

df = pd.concat([df_LeeG0, df_g1Z, df_g2Z])
[cm_s, scores_s, S_PredictAcetyl, S_PredictPhos] = CAROM_functions2.mdl_predict(
                          mdl=caromMdl_S,
                          X=df.loc[:,feature_names],
                          y=df.loc[:,"Target"],
                          condition="S_mdl", class_names=class_names,
                          gene_reactions=df[['genes','reaction','Phase']],
                          confusion_mat=True, bar=True, swarm=False,
                          pairwise=False, boxplot=False, explainer=None,
                          gscatter=False)

df = pd.concat([df_LeeG0, df_g1Z, df_sZ])
[cm_g2, scores_g2, G2_PredictAcetyl, G2_PredictPhos] = CAROM_functions2.mdl_predict(
                          mdl=caromMdl_G2,
                          X=df.loc[:,feature_names],
                          y=df.loc[:,"Target"],
                          condition="G2_mdl", class_names=class_names,
                          gene_reactions=df[['genes','reaction','Phase']],
                          confusion_mat=True, bar=True, swarm=False,
                          pairwise=False, boxplot=False, explainer=None,
                          gscatter=False)

