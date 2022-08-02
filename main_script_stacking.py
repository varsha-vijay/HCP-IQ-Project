''' Stacking from saved optimized models file'''

# Importing useful modules: 
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import nibabel
import time

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
    
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

r2_scores_stacked = []
corr_vec_stacked = []

start = time.time()

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


for kk in [1,2,4]:
     filename = str(kk) + 'K_rbf_RF_RMSE_optimized_models.sav'
     path = 'C:\\Users\\priya\\Documents\\MSc Extended Research Project\\hcpsetmodels\\optimised models\\'
     
     optimized_models = pickle.load(open(path+filename, 'rb'))
     
     for jj in range(1,6):
        print('SPLIT: ', jj)
        
        # Setting the matrices according to the split number.
        X_train , y_train , X_test , y_test , X_valid , y_valid = TTV_split(jj)
        
        # # Scale X_train and X_valid
        # scaler = StandardScaler().fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)
        # X_valid = scaler.transform(X_valid)
        
        # SCALING EACH ONE SEPARATELY.
        train_scaler = StandardScaler().fit(X_train)
        X_train = train_scaler.transform(X_train)
        
        test_scaler = StandardScaler().fit(X_test)
        X_test = test_scaler.transform(X_test)
        
        valid_scaler = StandardScaler().fit(X_valid)
        X_valid = valid_scaler.transform(X_valid)
        
        # create ranking model
        # ranker = Lasso()
        ranker = RandomForestRegressor(random_state = 42)
        # Maybe try RF as ranker.
        # Maybe try PCA followed by further feature selection.
        # Plot eigenvalue spread on plot, and observe the variance threshold.
        # Also try 32,64,128 for k.
        # PCA should give a better idea of how many features are meaningful.
        # save PCA eigenvectors --> take the mean and "step in the direction of the eigenvector"
        # prior to PCA perform Kbest to try removing noise. maybe split in half initially.
        selector = SelectFromModel(estimator = ranker , max_features = int(40320/kk),  threshold = -np.inf )
        
        selector.fit(X_train,y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        X_valid = selector.transform(X_valid)
    
        level0 = list()
    
        count = 0 
        
        for ii in np.arange(0+count,20+count,5):
            # print(i)
            # name = type(loaded_models[i]).__name__
            name = type(optimized_models[ii]).__name__
            
            if name == 'KernelRidge' :
                continue
            
            # print(name)
            # print(loaded_models[i])
            # level0.append((name,loaded_models[i]))
            level0.append((name,optimized_models[ii]))
            count = count + 1
            
        # define meta learner model
        level1 = LinearRegression()
        
        stacker = StackingRegressor(estimators = level0, final_estimator = level1 )
                
        stacker.fit(X_train,y_train)
        
        r2 = stacker.score(X_test,y_test)
        r2_scores_stacked.append((kk,jj,r2))
        
        corr = np.sqrt(r2)
        corr_vec_stacked.append((kk,jj,corr))
        
print('TIME TAKEN: ', time.time() - start )
# TIME TAKEN:  3274.9606132507324
# TIME TAKEN:  18620.40162706375

# TIME TAKEN:  9951.666650056839

#%%

# df_r2 = pd.DataFrame(r2_scores)
# df_corr = pd.DataFrame(corr_vec)

df_str = pd.DataFrame(r2_scores_stacked)
df_stcorr = pd.DataFrame(corr_vec_stacked)
