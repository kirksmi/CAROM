#!/usr/bin/env python
# coding: utf-8

"""
Created on Mon Feb 15 09:24:54 2021

@author: kirksmi
"""
from sklearn.ensemble import RandomForestClassifier
import xgboost

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import make_scorer
import scipy.stats as st
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from functions import CAROM_functions2
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from ismember import ismember
from sklearn.model_selection import cross_val_predict
import os



#%%
df_G0 = pd.read_csv("../data/LeeG0_Norm.csv")
df_G1 = pd.read_csv("../data/LeeG1_NormZ.csv")
df_S = pd.read_csv("../data/LeeS_NormZ.csv")
df_G2 = pd.read_csv("../data/LeeG2_NormZ.csv")

df_zscores = pd.read_csv("../data/LeeMaxZscores.csv")

#%%
feature_names = df_G0.columns[3:16]   # index version
print(feature_names)
# X = df_G0[feature_names]
# y = df_zscores["MaxZ_G0"]
#%%
# find genes in feature dataset that have z-score measurements
ix,pos = ismember(df_G0.genes, df_zscores.Genes.str.upper())
print(sum(ix))

# new DF with only measured genes
df_G0rev = df_G0[ix]
df_G0rev = df_G0rev.reset_index()

zscores = np.array(df_zscores.loc[pos, "MaxZ_G0"])
df_G0rev["zscore"] = zscores
#%%
### holdout validation model
X = df_G0rev[feature_names]
y = df_G0rev["zscore"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=123)


xgb_reg = xgboost.XGBRegressor(objective ='reg:squarederror',
                               colsample_bytree = 0.7, learning_rate = 0.1,
                               max_depth = 5, alpha = 10,
                               n_estimators = 150)
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))

plt.scatter(y_test, y_pred, c = "blue")
plt.title("Z-Score Regression Model")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.show()

# fit model on all data
xgb_reg.fit(X, y)

#%%
### cross-val predictions

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(xgb_reg, X=X, y=y, cv=5)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
#%%
### test G1

# revised G1 dataframe with measured genes only
ix, pos = ismember(df_G1.genes, df_zscores.Genes.str.upper())
df_G1rev = df_G1[ix].reset_index()
df_G1rev["zscore"] = np.array(df_zscores.loc[pos, "MaxZ_G1"])

# train and test vars
X_train = df_G0rev[feature_names]
y_train = df_G0rev["zscore"]
X_test = df_G1rev[feature_names]
y_test = df_G1rev["zscore"]

y_predG1 = xgb_reg.predict(df_G1rev[feature_names])
pltFont = {'fontname':'Arial'}
plt.rcParams.update(plt.rcParamsDefault) 


# define path name for saving trees
path = "../figures/regression/"

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)



fig, ax = plt.subplots(figsize=(8,6))
ax.set_axisbelow(True)
ax.grid(True,color='white')
ax.set_facecolor('lightgrey')
ax.scatter(y_test, y_predG1)
ax.set_xlabel('Measured Z-score', fontsize=18, **pltFont)
ax.set_ylabel('Predicted Z-score', fontsize=18, **pltFont)
ax.tick_params(axis = "both", which = "both", labelsize=16)
plt.title("Phosphorylation Z-score Model: G1 Phase", fontsize=20, **pltFont)

plt.tight_layout()
plt.savefig(path+"g1_regression_scatter.png",
            bbox_inches='tight', dpi=600)
plt.show()

print("Model prediction r^2: ",st.pearsonr(y_test, y_predG1))
print("G0 r^2: ",st.pearsonr(y_test, y_train))
#%%
### test G2

# make revised G2 dataset
ix, pos = ismember(df_G2.genes, df_zscores.Genes.str.upper())
df_G2rev = df_G2[ix].reset_index()
df_G2rev["zscore"] = np.array(df_zscores.loc[pos, "MaxZ_G2"])

# train and test vars
X_train = df_G0rev[feature_names]
y_train = df_G0rev["zscore"]
X_test = df_G2rev[feature_names]
y_test = df_G2rev["zscore"]

y_predG2 = xgb_reg.predict(X_test)

fig, ax = plt.subplots(figsize=(8,6))
ax.set_axisbelow(True)
ax.grid(True,color='white')
ax.set_facecolor('lightgrey')
ax.scatter(y_test, y_predG2)
ax.set_xlabel('Measured Z-score', fontsize=18, **pltFont)
ax.set_ylabel('Predicted Z-score', fontsize=18, **pltFont)
ax.tick_params(axis = "both", which = "both", labelsize=16)
plt.title("Phosphorylation Z-score Model: G2 Phase", fontsize=20, **pltFont)
plt.tight_layout()
plt.savefig(path+"g2_regression_scatter.png",
            bbox_inches='tight', dpi=600)
plt.show()


print("Model prediction r^2: ",st.pearsonr(y_test, y_predG2))
print("G0 r^2: ",st.pearsonr(y_test, y_train))

# print("numpy corrcoef: ",np.corrcoef(y_test, y_predG2)[0,1])
# print("pandas correlation: ", y_test.corr(pd.Series(y_predG2)))

### test S

# make revised G2 dataset
df_Srev = df_S[ix].reset_index()
df_Srev["zscore"] = np.array(df_zscores.loc[pos, "MaxZ_S"])

# train and test vars
X_train = df_G0rev[feature_names]
y_train = df_G0rev["zscore"]
X_test = df_Srev[feature_names]
y_test = df_Srev["zscore"]

y_predS = xgb_reg.predict(X_test)

fig, ax = plt.subplots(figsize=(8,6))
ax.set_axisbelow(True)
ax.grid(True,color='white')
ax.set_facecolor('lightgrey')
ax.scatter(y_test, y_predS)
ax.set_xlabel('Measured Z-score', fontsize=18, **pltFont)
ax.set_ylabel('Predicted Z-score', fontsize=18, **pltFont)
ax.tick_params(axis = "both", which = "both", labelsize=16)
plt.title("Phosphorylation Z-score Model: S Phase", fontsize=20, **pltFont)
plt.tight_layout()
plt.savefig(path+"s_regression_scatter.png",
            bbox_inches='tight', dpi=600)
plt.show()


print("Model prediction r^2: ",st.pearsonr(y_test, y_predS))
print("G0 r^2: ",st.pearsonr(y_test, y_train))

#%%
from yellowbrick.regressor import prediction_error
visualizer = prediction_error(xgb_reg, X_train, y_train, X_test, y_test)