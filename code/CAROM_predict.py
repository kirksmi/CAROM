#!/usr/bin/env python
# coding: utf-8

"""
Created on Mon Dec 28 15:10:25 2020

@author: kirksmi
"""
#%%
import pandas as pd

from CAROM_functions import caromPredict

#%%
# load dataset to set
df = pd.read_csv("../data/Lee_NormZ.csv")

featureNames = ['geneKO','maxATPafterKO','growthAcrossCond','closeness',
                'degree','betweenness','pagerank','reversible','rawVmin',
                'rawVmax','PFBAflux','kcat','MW']
df_test = df[featureNames]
print(df_test.head())
genes = df.genes
#%%
[df_predict, phosGenes, acetylGenes] = caromPredict(df_test, genes)



