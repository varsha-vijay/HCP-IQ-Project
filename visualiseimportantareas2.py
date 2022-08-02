
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

areas = [  4,   8,   9,  10,  12,  16,  18,  20,  24,  26,  31,  32,  33,
        34,  38,  39,  42,  45,  52,  56,  64,  70,  71,  83,  90,  92,
        94,  96,  97,  98,  99, 102, 117, 118, 119, 120, 122, 123, 125,
       130, 133, 139, 142, 143, 151, 154, 158, 160, 164, 166, 167, 168,
       169, 171, 174, 175, 183, 184, 186, 188, 190, 192, 195, 204, 219,
       220, 229, 232, 242, 252, 260, 261, 266, 268, 272, 274, 284, 285,
       286, 288, 290, 291, 296, 298, 304, 305, 306, 314, 316, 317, 320,
       324, 326, 328, 338, 343, 348, 349, 354, 356]

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