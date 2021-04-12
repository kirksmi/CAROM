#!/usr/bin/env python
# coding: utf-8

"""
# 'CAROM_Supplementary.py' contains the code for producing the majority of the supplementary
# models and figures described in the CAROM-ML paper. 
# 
# For the main figures and models, refer to 'CAROM_script.py'. Several other 
# supplementary analyses are carried out in the 'CAROM_zscoreRegression.py' and
# 'CAROM_algorithmComparison.py' files.
#
# The script is broken into the following sections:
# Part 1: Data import and setup
# Part 2: Training the supplementary ML models (e.g Yeast-only)
# Part 3: Explaining the organism-specific models using the SHAP package
# Part 4: Cell-cycle acetylation predictions using the no-G0 model
# Part 5: Validating the various cell-cycle models
#           -e.g caromMdl_G1 --> G0 data replaced w/ G1, then validated on
#            G0/S/G2
# Part 6: Validating the Phos-only (binary) model
#
#
# For details on using the custom functions in this script, use the help 
# function (e.g help(carom.xgb_func)).

@author: Kirk Smith
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from functions import carom 
import pickle
from collections import Counter
import xlsxwriter

#%%

### Part 1: Data import and setup ###

# primary dataset
df_carom = pd.read_csv("../Supplementary data/caromDataset.csv")
# create dataframes for each organism type
df_ecoli = df_carom.loc[df_carom.Organism==1]
df_yeast = df_carom.loc[df_carom.Organism==2]
df_human = df_carom.loc[df_carom.Organism==3]
# carom dataset with no cell-cycle data:
df_noLee = pd.read_csv("../Supplementary data/caromDataset_NoCellCycle.csv")
# dataset where background gene cross-reference is not applied:
df_noBackground =pd.read_csv("../Supplementary data/caromDataset_NoBackground.csv")
# cell cycle datasets:
df_g0Z = pd.read_excel("../Supplementary data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_G0")
df_g1Z = pd.read_excel("../Supplementary data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_G1")
df_sZ = pd.read_excel("../Supplementary data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_S")
df_g2Z = pd.read_excel("../Supplementary data/CellCycle_Scaled.xlsx",
                       sheet_name="CellCycle_G2")
df_CellCycle = pd.concat([df_g1Z,df_sZ,df_g2Z])

df_g1 = pd.read_csv("C:/Users/kirksmi/Documents/CAROM/final_datasets/LeeG1_Norm.csv")
df_s = pd.read_csv("C:/Users/kirksmi/Documents/CAROM/final_datasets/LeeS_Norm.csv")
df_g2 = pd.read_csv("C:/Users/kirksmi/Documents/CAROM/final_datasets/LeeG2_Norm.csv")

df_helaCholest = pd.read_csv("C:/Users/kirksmi/Documents/CAROM/final_datasets/HelaCholest_Norm.csv")
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

# view number of gene-rxn pairs by class
print("Number of regulated gene-rxn pairs:\n", df_carom['Target'].value_counts())
# view number of gene-rxn pairs by organism types
print("Number of each cell type:\n", df_carom['Organism'].value_counts())

#%%
### Feature correlation heatmaps

# create folder for saving figures
path = "../figures/correlation_heatmaps"
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

# plot correlation heatmaps
for df, condition in zip([df_ecoli[feature_names], df_yeast[feature_names],
                          df_human[feature_names], df_carom[featsPlusOrg]],
                         ["ecoli","yeast","human","allOrganisms"]):
    corrmat = df.corr()
    top_corr_features = corrmat.index
    # plot heat map
    plt.figure(figsize=(10,10))
    sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn",
                     vmin=-0.75, vmax=0.75,fmt='.2f')
    plt.savefig("../figures/correlation_heatmaps/{}_CorrHeatmap.png".format(condition),
                     bbox_inches='tight', dpi=600)

#%%

### Part 2: Train supplementary XGBoost models ###

# E. coli
[ecoli_mdl, ecoli_scores]= carom.xgb_func(
   X=df_ecoli[feature_names],
   y=df_ecoli['Target'],
   num_iter=5,
   condition='Ecoli_NoWeights', 
   class_names=class_names,
   imbalance="none")
filename1 = '../models/ecoli_XGBmodel.sav'
pickle.dump(ecoli_mdl, open(filename1, 'wb'))

# yeast
[yeast_mdl, yeast_scores]= carom.xgb_func(
   X=df_yeast[feature_names],
   y=df_yeast['Target'],
   num_iter=5,
   condition='Yeast_NoWeights', 
   class_names=class_names,
   imbalance="none")
filename1 = '../models/yeast_XGBmodel.sav'
pickle.dump(yeast_mdl, open(filename1, 'wb'))

# human
[human_mdl, human_scores]= carom.xgb_func(
   X=df_human[feature_names],
   y=df_human['Target'],
   num_iter=5,
   condition='Human_NoWeights', 
   class_names=class_names,
   imbalance="none")
filename1 = '../models/human_XGBmodel.sav'
pickle.dump(human_mdl, open(filename1, 'wb'))

# organism-type included as feature
[caromOrg_mdl, caromOrg_scores]= carom.xgb_func(
   X=df_carom[featsPlusOrg],
   y=df_carom['Target'],
   num_iter=5,
   condition='caromOrg_NoW', 
   class_names=class_names,
   imbalance="none")
filename1 = '../models/caromOrg_XGBmodel.sav'
pickle.dump(caromOrg_mdl, open(filename1, 'wb'))

# no cell-cycle data included
[caromMdl_NoG0, caromScores_NoG0]= carom.xgb_func(
   X=df_noLee[feature_names],
   y=df_noLee['Target'],
   num_iter=5,
   condition='NoG0', 
   class_names=class_names,
   imbalance="none")
filename1 = '../models/carom_XGBmodel_NoG0.sav'
pickle.dump(caromMdl_NoG0, open(filename1, 'wb'))

# no removal of non-annotated background genes
[caromMdl_background, caromScores_background]= carom.xgb_func(
   X=df_noBackground[feature_names],
   y=df_noBackground['Target'],
   num_iter=5,
   condition='NoBackground', 
   class_names=class_names,
   imbalance="none")
filename1 = '../models/carom_XGBmodel_NoBackground.sav'
pickle.dump(caromMdl_background, open(filename1, 'wb'))

## train models for the various cell-cycle phase datasets
# G1
df_caromG1 = pd.concat([df_noLee, df_g1Z])

[caromMdl_G1, caromScores_G1]= carom.xgb_func(
   X=df_caromG1[feature_names],
   y=df_caromG1['Target'],
   num_iter=5,
   condition='G1', 
   class_names=class_names,
   imbalance="none")
pickle.dump(caromMdl_G1, open('../models/carom_XGBmodel_G1.sav', 'wb'))

# S
df_caromS = pd.concat([df_noLee, df_sZ])

[caromMdl_S, caromScores_S]= carom.xgb_func(
   X=df_caromS[feature_names],
   y=df_caromS['Target'],
   num_iter=5,
   condition='S', 
   class_names=class_names,
   imbalance="none")
pickle.dump(caromMdl_S, open('../models/carom_XGBmodel_S.sav', 'wb'))

# G2
df_caromG2 = pd.concat([df_noLee, df_g2Z])

[caromMdl_G2, caromScores_G2]= carom.xgb_func(
   X=df_caromG2[feature_names],
   y=df_caromG2['Target'],
   num_iter=5,
   condition='G2', 
   class_names=class_names,
   imbalance="none")
pickle.dump(caromMdl_G2, open('../models/carom_XGBmodel_G2.sav', 'wb'))
#%%




# train model on G0 only

[caromMdl_onlyG0, caromScores_onlyG0]= carom.xgb_func(
   X=df_g0Z[feature_names],
   y=df_g0Z['Target'],
   num_iter=5,
   condition='onlyG0', 
   class_names=["Phos","Unreg"],
   imbalance="none")

[caromMdl_onlyG1, caromScores_onlyG1]= carom.xgb_func(
   X=df_g1Z[feature_names],
   y=df_g1Z['Target'],
   num_iter=5,
   condition='onlyG1', 
   class_names=["Phos","Unreg"],
   imbalance="none")

[caromMdl_onlyS, caromScores_onlyS]= carom.xgb_func(
   X=df_sZ[feature_names],
   y=df_sZ['Target'],
   num_iter=5,
   condition='onlyS', 
   class_names=["Phos","Unreg"],
   imbalance="none")

[caromMdl_onlyG2, caromScores_onlyG2]= carom.xgb_func(
   X=df_g2Z[feature_names],
   y=df_g2Z['Target'],
   num_iter=5,
   condition='onlyG2', 
   class_names=["Phos","Unreg"],
   imbalance="none")
#%%


# single phase models, no tuning

classifier = xgboost.XGBClassifier(objective='binary:logistic',
                    n_estimators=150,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=123)

classifier2 = xgboost.XGBClassifier(objective='multi:softprob',
                    n_estimators=150,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    num_class=3,
                    random_state=123)

classifier_onlyG0 = classifier.fit(X=df_g0Z[feature_names],
   y=pd.Series(le.transform(df_g0Z['Target'])))

classifier_onlyG1 = classifier.fit(X=df_g1Z[feature_names],
   y=pd.Series(le.transform(df_g1Z['Target'])))

classifier_onlyS = classifier.fit(X=df_sZ[feature_names],
   y=pd.Series(le.transform(df_sZ['Target'])))

classifier_onlyG2 = classifier.fit(X=df_g2Z[feature_names],
   y=pd.Series(le.transform(df_g2Z['Target'])))

classifier_onlyG1_fc = classifier.fit(X=df_g1[feature_names],
   y=pd.Series(le.transform(df_g1['Target'])))

classifer_caromG1 = classifier2.fit(df_caromG1[feature_names],
   y=pd.Series(le.transform(df_caromG1['Target'])))

#%%

# test one phase models


df = pd.concat([df_g1Z, df_sZ, df_g2Z])
[scores_g1, G1_PredictPhos, G1_ypred] = carom.mdl_predict(
                          mdl=caromMdl_onlyG0,
                          X=df[feature_names],
                          y=df["Target"],
                          condition="onlyG0",
                          class_names=["Phos","Unreg"],
                          gene_reactions=df[['genes','reaction','Phase']],
                          confusion_mat=True, bar=False)

df = pd.concat([df_g0Z, df_sZ, df_g2Z])
[scores_g1, G1_PredictPhos, G1_ypred] = carom.mdl_predict(
                          mdl=caromMdl_onlyG1,
                          X=df[feature_names],
                          y=df["Target"],
                          condition="onlyG1",
                          class_names=["Phos","Unreg"],
                          gene_reactions=df[['genes','reaction','Phase']],
                          confusion_mat=True, bar=False)

df = pd.concat([df_g0Z, df_g1Z, df_g2Z])
[scores_g1, G1_PredictPhos, G1_ypred] = carom.mdl_predict(
                          mdl=caromMdl_onlyS,
                          X=df[feature_names],
                          y=df["Target"],
                          condition="onlyS",
                          class_names=["Phos","Unreg"],
                          gene_reactions=df[['genes','reaction','Phase']],
                          confusion_mat=True, bar=False)

df = pd.concat([df_g0Z, df_g1Z, df_sZ])
[scores_g1, G1_PredictPhos, G1_ypred] = carom.mdl_predict(
                          mdl=caromMdl_onlyG2,
                          X=df[feature_names],
                          y=df["Target"],
                          condition="onlyG2",
                          class_names=["Phos","Unreg"],
                          gene_reactions=df[['genes','reaction','Phase']],
                          confusion_mat=True, bar=False)

#%%

## train G1 fold change models
df_caromG1 = pd.concat([df_noLee, df_g1])

[caromMdl_G1, caromScores_G1]= carom.xgb_func(
   X=df_caromG1[feature_names],
   y=df_caromG1['Target'],
   num_iter=5,
   condition='G1_FC', 
   class_names=class_names,
   imbalance="none")

[caromMdl_onlyG1, caromScores_onlyG1]= carom.xgb_func(
   X=df_g1[feature_names],
   y=df_g1['Target'],
   num_iter=5,
   condition='onlyG1_FC', 
   class_names=["Unreg","Phos"],
   imbalance="none")

#%%
df_sg2 = pd.concat([df_s, df_g2])

carom.mdl_predict(mdl=classifer_caromG1,
                X=df_sg2[feature_names],
                y=df_sg2["Target"],
                condition="testG1Mdl",
                gene_reactions=df_sg2[['genes','reaction','Phase']],
                confusion_mat=True, bar=True, swarm=False,
                pairwise=False, boxplot=False,
                explainer=None)



#%%
## train binary models (Unreg vs Acetyl, Unreg vs Phos)

# Acetyl
df_acetyl = pd.concat([df_carom[feature_names],df_carom.Acetyl],axis=1).dropna()

[AcetylMdl, AcetylScores]= carom.xgb_func(
   X=df_acetyl[feature_names],
   y=df_acetyl['Acetyl'],
   num_iter=5,
   condition='Acetyl', 
   class_names=["Unreg","Acetyl"],
   imbalance="none")
pickle.dump(AcetylMdl, open('../models/acetyl_XGBmodel.sav', 'wb'))

#Phos 
df_phos = pd.concat([df_carom[feature_names], df_carom.Phos],axis=1).dropna()

[PhosMdl_shallow, PhosScores_shallow]= carom.xgb_func(
   X=df_phos[feature_names],
   y=df_phos['Phos'],
   num_iter=5,
   condition='Phos', 
   class_names=["Unreg","Phos"],
   imbalance="none",
   depth='shallow')
pickle.dump(PhosMdl_shallow, open('../models/phosShallow_XGBmodel.sav', 'wb'))

 #%%
## load the models from disk
ecoli_mdl = pickle.load(open('../models/ecoli_XGBmodel.sav', 'rb'))
yeast_mdl = pickle.load(open('../models/yeast_XGBmodel.sav', 'rb'))
human_mdl = pickle.load(open('../models/human_XGBmodel.sav', 'rb'))
caromOrg_mdl = pickle.load(open('../models/caromOrg_XGBmodel.sav', 'rb'))
caromMdl_NoG0 = pickle.load(open('../models/carom_XGBmodel_NoG0.sav', 'rb'))
caromMdl = pickle.load(open('../models/carom_XGBmodel.sav', 'rb'))
phosMdl = pickle.load(open('../models/phosShallow_XGBmodel.sav', 'rb'))

#%%
### Part 3: SHAP analysis for various organism-type models

[ecoli_explainer, ecoli_shapValues] = carom.shap_func(
    xgbModel = ecoli_mdl,
    X = df_ecoli[feature_names],
    condition = "Ecoli",
    class_names = ["Phosphorylation","Unregulated","Acetylation"]) 

[yeast_explainer, yeast_shapValues] = carom.shap_func(
    xgbModel = yeast_mdl,
    X = df_yeast[feature_names],
    condition = "Yeast",
    class_names = ["Phosphorylation","Unregulated","Acetylation"])

[human_explainer, human_shapValues] = carom.shap_func(
    xgbModel = human_mdl,
    X = df_human[feature_names],
    condition = "Human",
    class_names = ["Phosphorylation","Unregulated","Acetylation"]) 

[caromOrg_explainer, caromOrg_shapValues] = carom.shap_func(
    xgbModel = caromOrg_mdl,
    X = df_carom[featsPlusOrg],
    condition = "CAROMwithOrg",
    class_names = ["Phosphorylation","Unregulated","Acetylation"]) 


#%%

### Part 4: Acetylation predictions for cell-cycle data with no-G0 model ###

[G0_acetylGenes, G0_phosGenes, G0_ypred] = carom.mdl_predict(
                                                    mdl=caromMdl_NoG0,
                                                    X=df_g0Z[feature_names],
                                                    condition="PredictG0_NoG0Mdl",
                                                    class_names=class_names,
                                                    gene_reactions=df_g0Z[['genes','reaction']])
G0_acetylGenes["Phase"] = np.tile("G0",(len(G0_acetylGenes),1))

[G1_acetylGenes, G1_phosGenes, G1_ypred] = carom.mdl_predict(
                                                    mdl=caromMdl_NoG0,
                                                    X=df_g1Z[feature_names],
                                                    condition="PredictG1_NoG0Mdl",
                                                    class_names=class_names,
                                                    gene_reactions=df_g1Z[['genes','reaction']])
G1_acetylGenes["Phase"] = np.tile("G1",(len(G1_acetylGenes),1))


[S_acetylGenes, S_phosGenes, S_ypred] = carom.mdl_predict(
                                                    mdl=caromMdl_NoG0,
                                                    X=df_sZ[feature_names],
                                                    condition="PredictS_NoG0Mdl",
                                                    class_names=class_names,
                                                    gene_reactions=df_sZ[['genes','reaction']])
S_acetylGenes["Phase"] = np.tile("S",(len(S_acetylGenes),1))


[G2_acetylGenes, G2_phosGenes, G2_ypred] = carom.mdl_predict(
                                                    mdl=caromMdl_NoG0,
                                                    X=df_g2Z[feature_names],
                                                    condition="PredictG2_NoG0Mdl",
                                                    class_names=class_names,
                                                    gene_reactions=df_g2Z[['genes','reaction']])
G2_acetylGenes["Phase"] = np.tile("G2",(len(G2_acetylGenes),1))
#%%

# combine predictions and write to file
LeeAcetylPred_AllPhases = pd.concat([G0_acetylGenes,G1_acetylGenes,
                                            S_acetylGenes, G2_acetylGenes])
LeePhosPred_AllPhases = pd.concat([G0_phosGenes,G1_phosGenes,
                                            S_phosGenes, G2_phosGenes])
LeeAcetylPred_AllPhases.to_csv('../results/CellCycleAcetylPred_AllPhases.csv',
                                index=False)

# get unique Acetyl genes and rxns
uniqueLeeAcetylGenes = LeeAcetylPred_AllPhases.genes.unique()
uniqueLeeAcetylRxns = LeeAcetylPred_AllPhases.reaction.unique()

uniqueLeePhosGenes = LeePhosPred_AllPhases.genes.unique()
#%%

array = [uniqueLeeAcetylGenes,
         uniqueLeeAcetylRxns]

workbook = xlsxwriter.Workbook('../results/CellCycleAcetylPred_Unique.xlsx')
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

### Part 5: Validating the various cell-cycle models ###

# G1 model
df = pd.concat([df_g0Z, df_sZ, df_g2Z])
[scores_g1, G1_PredictAcetyl, G1_PredictPhos, G1_ypred] = carom.mdl_predict(
                          mdl=caromMdl_G1,
                          X=df.loc[:,feature_names],
                          y=df.loc[:,"Target"],
                          condition="G1_mdl", class_names=class_names,
                          gene_reactions=df[['genes','reaction','Phase']],
                          confusion_mat=True, bar=True, swarm=False,
                          pairwise=False, boxplot=False,
                          explainer=None)

# S model
df = pd.concat([df_g0Z, df_g1Z, df_g2Z])
[scores_s, S_PredictAcetyl, S_PredictPhos, S_ypred] = carom.mdl_predict(
                          mdl=caromMdl_S,
                          X=df.loc[:,feature_names],
                          y=df.loc[:,"Target"],
                          condition="S_mdl", class_names=class_names,
                          gene_reactions=df[['genes','reaction','Phase']],
                          confusion_mat=True, bar=True, swarm=False,
                          pairwise=False, boxplot=False,
                          explainer=None)

# G2 model
df = pd.concat([df_g0Z, df_g1Z, df_sZ])
[scores_g2, G2_PredictAcetyl, G2_PredictPhos, G2_ypred] = carom.mdl_predict(
                          mdl=caromMdl_G2,
                          X=df.loc[:,feature_names],
                          y=df.loc[:,"Target"],
                          condition="G2_mdl", class_names=class_names,
                          gene_reactions=df[['genes','reaction','Phase']],
                          confusion_mat=True, bar=True, swarm=False,
                          pairwise=False, boxplot=False,
                          explainer=None)
#%%

# Part 6: Validating the binary Phos model ###

X = df_CellCycle.loc[:,feature_names]
y = df_CellCycle.loc[:,"Phos"]
print(y.value_counts())

[phosMdl_scores, phosMdl_PredictPhos, phosMdl_Ypred] = carom.mdl_predict(
                          mdl=PhosMdl_shallow, X=X, y=y,
                          condition="PhosMdl", class_names=["Unreg","Phos"],
                          gene_reactions=df_CellCycle[['genes','reaction','Phase']],
                          confusion_mat=True, bar=True, swarm=False,
                          pairwise=False, boxplot=False,
                          explainer=None)
                          


