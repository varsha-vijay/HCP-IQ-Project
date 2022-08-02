# Run this file after running the first visualise_important_areas.py file to extract the top 100 regions. 

#%% IMPORTING BASIC MODULES
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import nibabel
import time
import copy

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso


# %% Now we need to display only the relevant important areas.
labelL = nibabel.load('C:\\Users\\priya\\Documents\\MSc Extended Research Project\\Trial Labels\\100307.L.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')
labelR = nibabel.load('C:\\Users\\priya\\Documents\\MSc Extended Research Project\\Trial Labels\\100307.R.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')
    
label_names_L = labelL.labeltable.get_labels_as_dict()
label_names_R = labelR.labeltable.get_labels_as_dict()

labelLimportant = copy.deepcopy(labelL)
labelRimportant = copy.deepcopy(labelR)

# Add the top 100 regions identified by the previous file below as csv list of ints
areas = [ ]

for i in np.arange(1,len(label_names_L)):
    label_pos_L = np.where(labelL.darrays[0].data == i )[0]
    if i not in areas:
        labelLimportant.darrays[0].data[label_pos_L] = 0

for i in np.arange(1,len(label_names_R)):
    label_pos_R = np.where(labelR.darrays[0].data == i)[0]
    if i not in areas:
        labelRimportant.darrays[0].data[label_pos_R] = 0

# Now saving the files:
nibabel.save(labelLimportant, 'Limpo_features_mean.func.gii')
nibabel.save(labelRimportant, 'Rimpo_features_mean.func.gii')
