# This is a file to extract some confounding variables: 

# First need to import useful libraries: 
import numpy as np
import nibabel 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import copy

# Defining a function to extract the ages: 
def intercranialvol_extractor(sheet_num):
    
    # Sheet_num: represents whether to load in training or testing data.
    
    # Getting the age data:
    info_file_name = 'C:\\Users\\priya\\Documents\\MSc Extended Research Project\\Part 2 Data\\unrestricted_hcp_freesurfer (1).csv'
    df = pd.read_csv(info_file_name)
    info_data = df.to_numpy()
    
    file_name = 'C:\\Users\\priya\\Documents\\MSc Extended Research Project\\Data\\Crossval Splits.xlsx'
    df = pd.read_excel(io = file_name, sheet_name = sheet_num)
    data  = df.to_numpy()
    
    Feature_filenames_L = data[:,0]
    
    # ALL the cases - ie all the subjects so 1206 different ones
    cases = info_data[:,0]
    # Converting cases from string to int
    for k in range(len(cases)):
          int(cases[k])
    
    
    # GETTING THE RELEVANT DATA. We only want some of the subjects for each split
    relevant_cases = []
    for m in range(len(Feature_filenames_L)):
        
        # Extracting case number. This extracts just the id of the necessary subjects per split
        # and stores them in an array.
        string = Feature_filenames_L[m]
        string1 = string.split(".")
        relevant_cases.append(int(string1[0]))
    
    
    # Getting the location index of relevant cases
    ind = []
    for i in range(len(relevant_cases)):
        for j in range(len(cases)):
            if(relevant_cases[i] == cases[j]):
                ind.append(j)  
                
    # Extracting weight info of relevant cases
    y_out = np.zeros(len(ind))
    for i in range(len(ind)):
        y_out[i] = info_data[:,3][ind[i]]
    
    
    return y_out
    
    

# Now:
print("Running the function to create intercranial volume_splits: ")

sheet_num= ['Split 1', 'Split 2', 'Split 3', 'Split 4', 'Split 5']
X_names = ['icvol_split1', 'icvol_split2', 'icvol_split3','icvol_split4', 'icvol_split5']

for i, s in enumerate(sheet_num):
    X_out = intercranialvol_extractor(s)
    np.save(X_names[i], X_out)

