# Importing relevant files: 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel
import time

# File to create the x splits:
    
def X_maker_splits(sheet_num):
    # Generates matrices containing rows = number of samples
    # columns = features and labels
    
    # sheet_num: represents which split to load in
    # colL: column number of column containing features of L
    # colR: column number of column containing features of R
    
    # Outputs X_TRAIN< X_TEST or X_VALID depending on the sheet num
    
    # file_name = #path to file + # file name
    file_name = 'C:\\Users\\priya\\Documents\\MSc Extended Research Project\\Data\\Crossval Splits.xlsx'
    
    df = pd.read_excel(io=file_name, sheet_name = sheet_num)
    
    # sheet num represents which split is loaded in
    data = df.to_numpy()
    
    # Extract the Features_filenames for left and right hemisphere
    Feature_filenames_L = data[:,0]
    Feature_filenames_R = data[:,1]
    
    Labels_filenames_L = data[:,2]
    Labels_filenames_R = data[:,3]
    
    path_labels = 'C:\\Users\\priya\\Documents\\MSc Extended Research Project\\alllabels\\'
    path_features = 'C:\\Users\\priya\\Documents\\MSc Extended Research Project\\allfeatures\\'
    
    X_out = np.zeros([180*112*2])
    
    for m in range(len(Feature_filenames_L)):
        
        # Loading up
        featuresL = nibabel.load(path_features+Feature_filenames_L[m])
        featuresR = nibabel.load(path_features+Feature_filenames_R[m])
        
        labelL = nibabel.load(path_labels+Labels_filenames_L[m])
        labelR = nibabel.load(path_labels+Labels_filenames_R[m])
        
        label_names_L = labelL.labeltable.get_labels_as_dict()
        label_names_R = labelR.labeltable.get_labels_as_dict()
        np.array([0])
        
        if (len(label_names_L)==181):
            row = []
            for i in np.arange(1,len(label_names_L)):
                for j in range(featuresL.numDA):
                    # Variable to give location of each label.
                    label_pos_L = np.where(labelL.darrays[0].data == i )[0]
                    
                    # fj_li_L = featuresL.darrays[j].data[label_pos_L]                 
                    # val_L = fj_li_L[0]
                    
                    temp_mean_L = np.mean(featuresL.darrays[j].data[label_pos_L])
                    # temp_mean_L = np.mean(abs(featuresL.darrays[j].data[label_pos_L]))
                    # temp_std_L = np.std(featuresL.darrays[j].data[label_pos_L])
        
                    label_pos_R = np.where(labelR.darrays[0].data == i )[0]
                    
                    temp_mean_R = np.mean(featuresR.darrays[j].data[label_pos_R])
                    # temp_mean_R = np.mean(abs(featuresR.darrays[j].data[label_pos_R]))
                    # temp_std_R = np.std(featuresR.darrays[j].data[label_pos_R])
        
                    # fj_li_R = featuresR.darrays[j].data[label_pos_R] 
                    # val_R = fj_li_R[0]
        
                    # FOR MEAN
                    row.append(temp_mean_L)
                    row.append(temp_mean_R)
                    
                    # FOR STD 
                    # row.append(temp_std_L)
                    # row.append(temp_std_R)
                
        if (len(label_names_L) == 361):   
            row = [] 
            for i in np.arange(1,len(label_names_L)/2):
                for j in range(featuresL.numDA):
                # for j in range(100): # For removing the last 12 features.
                        
                    # Variable to give location of each label.
                    label_pos_L = np.where(labelL.darrays[0].data == (180+i) )[0]
                    
                    # fj_li_L = featuresL.darrays[j].data[label_pos_L]                 
                    # val_L = fj_li_L[0]
                    
                    temp_mean_L = np.mean(featuresL.darrays[j].data[label_pos_L])
                    # temp_mean_L = np.mean(abs(featuresL.darrays[j].data[label_pos_L]))
                    # temp_std_L = np.std(featuresL.darrays[j].data[label_pos_L])
        
                    label_pos_R = np.where(labelR.darrays[0].data == i )[0]
                    
                    temp_mean_R = np.mean(featuresR.darrays[j].data[label_pos_R])
                    # temp_mean_R = np.mean(abs(featuresR.darrays[j].data[label_pos_R]))
                    # temp_std_R = np.std(featuresR.darrays[j].data[label_pos_R])
        
                    # fj_li_R = featuresR.darrays[j].data[label_pos_R] 
                    # val_R = fj_li_R[0]
        
                    # FOR MEAN
                    row.append(temp_mean_L)
                    row.append(temp_mean_R)
                    
                    # FOR STD
                    # row.append(temp_std_L)
                    # row.append(temp_std_R)


        X_out = np.vstack([X_out,row])
                          
        print('Percent Completed:' ,(m*100)/len(Feature_filenames_L))
    
    X_out = np.delete(X_out, (0), axis=0)
    
    return X_out

# Calling the function: 

print("Running the function to create X_splits: ")

sheet_num= ['Split 1', 'Split 2', 'Split 3', 'Split 4', 'Split 5']
X_names = ['X_split1', 'X_split2', 'X_split3','X_split4', 'X_split5']

for i, s in enumerate(sheet_num):
    X_out = X_maker_splits(s)
    np.save(X_names[i], X_out)
    


                    