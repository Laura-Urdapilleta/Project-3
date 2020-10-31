#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:58:52 2020

@author: lauraurdapilleta
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Get pandas and postgres to work together

import numpy as np




#cross-validation
from sklearn.model_selection import train_test_split


# the model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree



#reports
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


# Full progress bar during long loop:
import sys


X = pd.read_csv('train_values.csv').drop(columns = 'building_id')
y = pd.read_csv('train_labels.csv').drop(columns = 'building_id')

_, X, _, y = train_test_split( X, y, stratify=y,
                              test_size=30000, random_state=42)

X = pd.get_dummies(X)
featurenames =X.columns

X.to_csv()
y.to_csv()

X_train, X_test, y_train, y_test = train_test_split( X, y, stratify=y,
                              test_size=.2, random_state=42)

#%%
from sklearn.ensemble import ExtraTreesClassifier

forest_selection = ExtraTreesClassifier(n_estimators=250, random_state=0)

forest_selection.fit(X_train, y_train)
importances = forest_selection.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest_selection.estimators_], axis=0)
indices = np.argsort(importances)[::-1]


#%%
number_selected_features = 6
subset = []
for i in indices[:number_selected_features]:
    subset.append(featurenames[i])
    
    
#%%
fig, ax = plt.subplots()

ax.bar(np.arange(number_selected_features), importances[indices[:number_selected_features]])


ax.set_title('Feature Importances')
ax.set_xticks(np.arange(number_selected_features))
ax.set_xticklabels(subset, rotation=40)
st.pyplot(fig)

#%%
X_train, X_test = X_train[subset], X_test[subset]

#%%
st.write(
'''
## Predicting with the selected model
'''
) 
selected_model = tree.DecisionTreeClassifier(random_state = 42, max_depth=6)
selected_model.fit(X_train, y_train)
#%%
# probabilities = selected_model.predict_proba(X_test)
# ypred = selected_model.predict(X_test)


geo_level_1_id = st.number_input('geo level id 1', value=7)
geo_level_3_id = st.number_input('geo level id 3', value=2313)
age = st.number_input('age', value=5)
geo_level_2_id = st.number_input('geo level id 2', value=617)
area_percentage = st.number_input('area percentage', value=8)
height_percentage = st.number_input('height percentage', value=5)


input_data = pd.DataFrame({
                            'geo level id 1': [geo_level_1_id],
                            'geo level id 3': [geo_level_3_id],
                            'age': [age],
                            'geo level id 2': [geo_level_2_id],
                            'area percentage': [area_percentage],
                            'height percentage': [height_percentage]
                            })
                            

probabilities = selected_model.predict_proba(input_data)[0]


fig, ax = plt.subplots()

ax.bar(np.arange(3), probabilities)


ax.set_title('Class Probabilities')
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['Low', 'medium', 'Destructed'], rotation=40)
# ax.axis()
st.pyplot(fig)


