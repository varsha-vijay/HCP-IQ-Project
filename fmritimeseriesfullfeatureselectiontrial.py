''' Full Feature Selection for rfMRI trials'''

# Now trialling random forest regressor after applying feature selection: 
#%% IMPORTING BASIC MODULES
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import nibabel
import time

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

# %% Load the features and PMAT scores: 

X_split1= np.load('timeseries1_X.npy')
X_split2= np.load('timeseries2_X.npy')
X_split3= np.load('timeseries3_X.npy')
X_split4= np.load('timeseries4_X.npy')
X_split5= np.load('timeseries5_X.npy')

y_split1 = np.load('ts_pmat1.npy')
y_split2 = np.load('ts_pmat2.npy')
y_split3 = np.load('ts_pmat3.npy')
y_split4 = np.load('ts_pmat4.npy')
y_split5 = np.load('ts_pmat5.npy')

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


#%% FUNCTION FOR HYPERPARAMETER OPTIMIZATION USING MANUAL COMBINATIONS

# Needed for functions to work.
# -----------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
from scipy.stats import pearsonr
# -----------------------------------------------------

def hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid):
    '''
    The main function performing manaual gridsearch for optimal hyperparameters.
    inputs = (model,X_train,y_train,X_valid,y_valid)
    outputs = (opti_param1, opti_param2)
    '''
    # Getting the name of the model.
    name = type(model).__name__
    
    graph_x = []
    graph_y = []
    graph_z = []
    
     # Lasso Functionality
    if name == 'Lasso' or name == 'Ridge' :
        for alpha_value in np.arange(-5.0,2.5,0.5):
            alpha_value = pow(10,alpha_value)
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []

            hyperparams = alpha_value
            rmse = Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid)
            graph_x_row.append(alpha_value)
            graph_z_row.append(rmse)
    
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)
            print('')
            
            
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        opti_alpha = graph_x[pos_min_z[0], pos_min_z[1]]
        # opti_gamma = graph_y[pos_min_z[0], pos_min_z[1]]
        print('Minimum RMSE: %.4f' %(min_z))
        # print('Optimum alpha: %f' %(graph_x[pos_min_z[0],pos_min_z[1]]))
        # print('Optimum gamma: %f' %(graph_y[pos_min_z[0],pos_min_z[1]]))
        print('Optimum alpha: %f' %(opti_alpha))
        # print('Optimum gamma: %f' %(opti_gamma))
        return opti_alpha 
    
    # RandomForestRegressor Functionality
    if name == 'RandomForestRegressor':
        
        array_feats = [100,250,500,750,1000,1250,1500]
        if X_train.shape[1] <= 1260 :
            array_feats = [100,150,200,250,300]
        
        # Below for PCA 
        # array_feats = [1]
        for max_feats in array_feats:
            # n_estim = pow(100,n_estim)
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []
            
            for min_samp_leaf in np.arange(5,25,5):
                hyperparams = (max_feats,min_samp_leaf)
                rmse = Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid)
                graph_x_row.append(max_feats)
                graph_y_row.append(min_samp_leaf)
                graph_z_row.append(rmse)
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)
            print('')
            
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        opti_feats = graph_x[pos_min_z[0], pos_min_z[1]]
        opti_samps = graph_y[pos_min_z[0], pos_min_z[1]]
        print('Minimum RMSE: %.4f' %(min_z))
        print('Optimum max features: %f' %(opti_feats))
        print('Optimum min_samples_leaf: %f' %(opti_samps))
        return opti_feats , opti_samps
    
    # AdaBoostRegressor Functionality
    if name == 'AdaBoostRegressor':
        for n_estims in [50,75,100,125,150]:
            # n_estim = pow(100,n_estim)
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []
            
            for loss in ['linear','square','exponential']:
                hyperparams = (n_estims,loss)
                rmse = Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid)
                graph_x_row.append(n_estims)
                graph_y_row.append(loss)
                graph_z_row.append(rmse)
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)
            print('')
            
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        opti_estims = graph_x[pos_min_z[0], pos_min_z[1]]
        opti_loss = graph_y[pos_min_z[0], pos_min_z[1]]
        print('Minimum RMSE: %.4f' %(min_z))
        print('Optimum num estimators: %f' %(opti_estims))
        print('Optimum loss: %s' %(opti_loss))
        return opti_estims , opti_loss
    
    if name == 'GradientBoostingRegressor':
        
        array_feats = [100,250,500,750,1000,1250,1500]
        if X_train.shape[1] <= 1260 :
            array_feats = [100,150,200,250,300]
        
        # Below for PCA 
        # array_feats = [1]
        
        for max_feats in array_feats:
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []
            
            for min_samp_leaf in np.arange(5,25,5):
                hyperparams = (max_feats,min_samp_leaf)
                rmse = Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid)
                graph_x_row.append(max_feats)
                graph_y_row.append(min_samp_leaf)
                graph_z_row.append(rmse)
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)
            print('')
            
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        opti_feats = graph_x[pos_min_z[0], pos_min_z[1]]
        opti_samps = graph_y[pos_min_z[0], pos_min_z[1]]
        print('Minimum RMSE: %.4f' %(min_z))
        print('Optimum max features: %f' %(opti_feats))
        print('Optimum min samp leaf: %f' %(opti_samps))
        return opti_feats , opti_samps
    
    
    
# %% New function: 
    
def Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid):
    '''
    Sub function used in hyperparams_gridsearch for calculating error metric for a certain combination of parameters.
    '''
    # Getting the name of the model.
    name = type(model).__name__
    print(name)

    # Feed X_VALID and y_valid to actually check each  combination's result.

    # Maybe try feature selection here with Kbest or feature importance from Random forest.
    if name == 'RandomForestRegressor':
        # Assign hyper-parameters
        max_feats,min_samp_leaf = hyperparams
        model.set_params(n_estimators = 100, max_features = max_feats , min_samples_leaf = min_samp_leaf, random_state = 42)
    
    if name == 'Ridge' or name == 'Lasso':
        # Assign hyperparameters
        alpha_value = hyperparams
        model.set_params(alpha = alpha_value)
    
    
    if name == 'GradientBoostingRegressor':
        # Assign hyper-parameters
        max_feats,min_samp_leaf = hyperparams
        model.set_params(n_estimators = 100,  max_features = max_feats ,min_samples_leaf = min_samp_leaf, random_state = 42)
        # below for PCA --> removing max_features parameter
        # model.set_params(n_estimators = 100,  min_samples_leaf = min_samp_leaf, random_state = 42) 
        
    if name == 'AdaBoostRegressor':
        # Assign hyper-parameters
        n_estims,loss = hyperparams
        model.set_params(n_estimators = n_estims, loss = loss ,random_state = 42)
            
        
    model.fit(X_train,y_train)
    y_pred  = model.predict(X_valid)
    err = mean_absolute_error(y_valid, y_pred)
    
    if name == 'RandomForestRegressor':
        print('Random Forest Regressor. Max_feats: %7.6f, Min_samps: %7.4f, RMSE: %7.4f' %(max_feats, min_samp_leaf ,err))
    
    if name == 'Ridge' or name ==  'Lasso':
        print('alpha: %7.6f, RMSE: %7.4f' %(alpha_value,err))
    
    if name == 'AdaBoostRegressor':
        print('AdaBoost Regressor. n_estims: %7.6f, loss: %s, RMSE: %7.4f' %(n_estims, loss ,err))
    
    if name == 'GradientBoostingRegressor':
        print('Gradient Boosting Regressor. Max_feats: %7.6f, Min_samps: %7.4f, RMSE: %7.4f' %(max_feats, min_samp_leaf ,err))

        
    return err

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

r2_scores = [] 
corr_vec = [] 
k = []

r2_scores_stacked = []
corr_vec_stacked = []

# These store the optimal params for each model on each split.
opti_alphas = []
opti_gammas = []

# Creating the Random Forest Regressor
models = list()
models.append(Lasso())
models.append(RandomForestRegressor())
models.append(AdaBoostRegressor())
models.append(GradientBoostingRegressor())
# name = type(model).__name__

# Trying for a different number of features selected: 
for kk in [2,4,8]:
    
    optimized_models = [] 
    
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
        ranker = RandomForestRegressor(random_state = 42)
        
        selector = SelectFromModel(estimator = ranker , max_features = int(64620/kk),  threshold = -np.inf )
        
        selector.fit(X_train,y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        X_valid = selector.transform(X_valid)
        
        for model in models:
            name = type(model).__name__
            print(name)
            
            if name == 'Lasso':
                # Call hyperparams_gridsearchfunction on X_train and y_train
                opti_alpha = hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid)
                # Making optimized model
                opti_model = model.set_params(alpha = opti_alpha)
                # Storing optimized model
                optimized_models.append(opti_model)
                
                # Storing optimal hyperparameters for each split.
                opti_alphas.append((kk,name,i, opti_alpha))
        
            if name == 'RandomForestRegressor':
                # Call hyperparams_gridsearchfunction on X_train and y_train
                opti_max_feats, opti_min_samps = hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid)
                # Making optimized model
                opti_model = model.set_params(n_estimators = 100 , max_features = opti_max_feats, min_samples_leaf = opti_min_samps, random_state = 42)
                # Storing optimized model
                optimized_models.append(opti_model)
                
                # Storing optimal hyperparameters for each split.
                opti_alphas.append((kk,name,i, opti_max_feats))
                opti_gammas.append((kk,name,i, opti_min_samps))
            
            if name == 'AdaBoostRegressor':
                # Call hyperparams_gridsearchfunction on X_train and y_train
                opti_estims, opti_loss = hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid)
                # Making optimized model
                opti_model = model.set_params(n_estimators = opti_estims, loss = str(opti_loss) ,random_state = 42)
                # Storing optimized model
                optimized_models.append(opti_model)
                
                # Storing optimal hyperparameters for each split.
                opti_alphas.append((kk,name,i, opti_estims))
                opti_gammas.append((kk,name,i, opti_loss))
            
            if name == 'GradientBoostingRegressor':
                # Call hyperparams_gridsearchfunction on X_train and y_train
                opti_max_feats, opti_min_samps = hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid)
                # Making optimized model
                opti_model = model.set_params(n_estimators = 100 , max_features = opti_max_feats, min_samples_leaf = opti_min_samps, random_state = 42)
                # Storing optimized model
                optimized_models.append(opti_model)
                
                # Storing optimal hyperparameters for each split.
                opti_alphas.append((kk,name,i, opti_max_feats))
                opti_gammas.append((kk,name,i, opti_min_samps))
        
            # Fitting the optimized model on the X_train.
            opti_model.fit(X_train,y_train)
                
            # Predict using optimized model on X_test
            r2 = opti_model.score(X_test,y_test)
            # Storing score on this split.
            r2_scores.append((kk,name,i, r2))
            
            corr = np.sqrt(r2)
            corr_vec.append((kk,name,i, corr))
            
            k.append(kk)

    # SAVING LIST OF OPTIMIZED MODELS
    filename = str(kk) +'K_rbf_RF_RMSE_optimized_models.sav'
    path = 'C:\\Users\\priya\\Documents\\MSc Extended Research Project\\optimisedmodelsforrfmri'

    save_name = path +filename
    pickle.dump(optimized_models, open(save_name, 'wb'))
