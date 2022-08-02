
#%% IMPORTING BASIC MODULES
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import nibabel
import time

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

 
# %% LOADING IN DATASETS
X_split1= np.load('X_split1.npy')
X_split2= np.load('X_split2.npy')
X_split3= np.load('X_split3.npy')
X_split4= np.load('X_split4.npy')
X_split5= np.load('X_split5.npy')

y_split1 = np.load('PMAT_y_split1.npy')
y_split2 = np.load('PMAT_y_split2.npy')
y_split3 = np.load('PMAT_y_split3.npy')
y_split4 = np.load('PMAT_y_split4.npy')
y_split5 = np.load('PMAT_y_split5.npy')

# Dealing with NaN and infinite values.
X_split1 = np.nan_to_num(X_split1)
X_split2 = np.nan_to_num(X_split2)
X_split3 = np.nan_to_num(X_split3)
X_split4 = np.nan_to_num(X_split4)
X_split5 = np.nan_to_num(X_split5)


#%% FUNCTION FOR TTV SPLIT
def TTV_split(split_num):
    """A function that sets the train, test, and validation matrices according to split number."""
    if split_num == 1:
        X_train = np.concatenate((X_split3,X_split4,X_split5)) 
        y_train = np.concatenate((y_split3,y_split4,y_split5)) 
        
        X_test = X_split1
        y_test = y_split1
        
        X_valid = X_split2
        y_valid = y_split2
    
    
    if split_num == 2:
        X_train = np.concatenate((X_split1,X_split4,X_split5)) 
        y_train = np.concatenate((y_split1,y_split4,y_split5))
        
        X_test = X_split2
        y_test = y_split2
        
        X_valid = X_split3
        y_valid = y_split3
        
    if split_num == 3:
        X_train = np.concatenate((X_split1,X_split2,X_split5)) 
        y_train = np.concatenate((y_split1,y_split2,y_split5)) 
        
        X_test = X_split3
        y_test = y_split3
                
        X_valid = X_split4
        y_valid = y_split4
    
    if split_num == 4:
        X_train = np.concatenate((X_split1,X_split2,X_split3)) 
        y_train = np.concatenate((y_split1,y_split2,y_split3)) 
        
        X_test = X_split4
        y_test = y_split4
        
        X_valid = X_split5
        y_valid = y_split5
        
    if split_num == 5:
        X_train = np.concatenate((X_split2,X_split3,X_split4)) 
        y_train = np.concatenate((y_split2,y_split3,y_split4))
        
        X_test = X_split5
        y_test = y_split5
        
        X_valid = X_split1
        y_valid = y_split1
        
    return X_train , y_train , X_test , y_test , X_valid , y_valid

# %%
impo = []

for i in range(1,6):
        print('SPLIT: ', i)
        # Setting the matrices according to the split number.
        X_train , y_train , X_test , y_test , X_valid , y_valid = TTV_split(i)
        
        # Scale X_train, X_test and X_valid accordin to X_train
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_valid = scaler.transform(X_valid)

        # FEATURE SELECTION BY SELECTING A CERTAIN NUMBER OF BEST FEATURES.
        # Create ranking model manually choose Lasso or Random forest regressor ranker
        # ranker = Lasso()
        model = RandomForestRegressor(random_state = 42)
        
        model.fit(X_train,y_train)
        importances = model.feature_importances_
        impo.append(importances)

mean_impos = np.mean(impo, axis = 0)

# reshape into 360 areas and 112 maps
reshaped_impos = np.reshape(mean_impos,(360,112))
impos_mean = np.mean(reshaped_impos, axis = 1)

reshaped_impos_mean = impos_mean[:]
reshaped_impos_mean = np.ndarray.tolist(reshaped_impos_mean)

# Now for the feature selection bit:
max_ind = []
maxes = []

for i in range(100):
    temp_max = max(reshaped_impos_mean)
    maxes.append(temp_max)
    temp_ind = reshaped_impos_mean.index(temp_max)
    max_ind.append(temp_ind)
    reshaped_impos_mean[temp_ind] = 0
    
areas = np.sort(max_ind)

# Now extracting the most important areas for the left and right:

# %% Now we need to display only the relevant important areas.
label_L = nibabel.load('C:\\Users\\priya\\Documents\\MSc Extended Research Project\\Trial Labels\\100307.L.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')
label_R = nibabel.load('C:\\Users\\priya\\Documents\\MSc Extended Research Project\\Trial Labels\\100307.R.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')
    


