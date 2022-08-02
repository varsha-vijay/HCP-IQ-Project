#!/usr/bin/env python


''' FINAL Script for functional connectivity data '''

# Importing relevant files: 
import numpy as np
import pandas as pd
import nibabel
import time

# %% fMRI timeseries analysis

# Folder path for all the extracted .gii timeseries.
#path_ts = '~/Documents/MSc\ Project\:\ HCP\ IQ/alltimeseries/'
path_ts = './'
# Folder path for all the subject labels. 
path_label = '~/Documents/MSc\ Project\:\ HCP\ IQ/alllabels/'

# List in order of all the required subjects.
subjects = ['100307','100408','101006','148032','102008','137633']

# The left hemisphere timeseries
Lfilenames = ['.L.rfMRI_REST2_RL_Atlas_hp2000_clean_vn.func.gii',
         '.L.rfMRI_REST2_LR_Atlas_hp2000_clean_vn.func.gii',
         '.L.rfMRI_REST1_RL_Atlas_hp2000_clean_vn.func.gii',
         '.L.rfMRI_REST1_LR_Atlas_hp2000_clean_vn.func.gii']

# The right hemisphere timeseries
Rfilenames =  ['.R.rfMRI_REST2_RL_Atlas_hp2000_clean_vn.func.gii',
         '.R.rfMRI_REST2_LR_Atlas_hp2000_clean_vn.func.gii',
         '.R.rfMRI_REST1_RL_Atlas_hp2000_clean_vn.func.gii',
         '.R.rfMRI_REST1_LR_Atlas_hp2000_clean_vn.func.gii',]

# The end of the label filenames.
label_filename_L = '.L.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii'
label_filename_R = '.R.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii'

# Initialising an array to hold the final upper triangle flattened for each subject
all_X = []

# Looping over each subject. 
for p,subject in enumerate(subjects):
    print('Beginning timeseries analysis for subject: ', p)
    
    main_one = []

    for m, name in enumerate(Lfilenames):
        print("Loading in rfMRI file : ", subject, name)
        
        # Load in each timeseries for the subject
        data = nibabel.load(subject + name)
	print('LOAD OF RESTING STATE FOLDER OKAY')
        
        # Load in the respective label for each subject
        label_L = nibabel.load('../alllabels/' + subject + label_filename_L)
	print('The LABEL IS GOOD TOO')
        # Extract label names.
        label_names_L = label_L.labeltable.get_labels_as_dict()
        
        test_row = []
        
        for i in np.arange(1,len(label_names_L)):
            temp = []
            for j in range(data.numDA): # each j corresponds to a different time point
                 # Variable to give location of each label.
		 label_pos_L = np.where(label_L.darrays[0].data == i )[0] # each 
		 if (len(label_names_L)==361):
                      label_pos_L = np.where(label_L.darrays[0].data ==	180+i)[0]	
                 #one of the 180 areas
                 
                 # Getting the mean timeseries over the area
                 # print(data.darrays[j].data[label_pos_L])
                 temp_mean_L = np.mean(data.darrays[j].data[label_pos_L])
                 temp.append(temp_mean_L)
            test_row.append(temp)
        
        main_one.append(test_row)
        
    # Adjusted version: 
    timeseries = []
    
    for x in range(180):
        temp = []
        temp = np.hstack([main_one[0][x], main_one[1][x], main_one[2][x], main_one[3][x]])
        timeseries.append(temp)
        
    
    # %% Right side:         
    print("Moving onto the right hemisphere.")
    
    main_one = []
    
    for m, name in enumerate(Rfilenames):        
        print("Loading in rfMRI file : ", subject, name)
        data = nibabel.load(subject + name)
        
        label_L = nibabel.load('../alllabels/' + subject + label_filename_R)
        #label_L = nibabel.load('C:\\Users\\priya\\Documents\\MSc Extended Research Project\\Trial Labels\\100307.L.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')
        
        label_names_L = label_L.labeltable.get_labels_as_dict()
        
        row = []
        test_row = []
        
        for i in np.arange(1,len(label_names_L)):
            temp = []
            for j in range(data.numDA): # each j corresponds to a different time point
                 # Variable to give location of each label.
                 label_pos_L = np.where(label_L.darrays[0].data == i )[0] # each one of the 180 areas
                 
                 # Getting the mean timeseries over the area
                 temp_mean_L = np.mean(data.darrays[j].data[label_pos_L])
                 temp.append(temp_mean_L)
        
            test_row.append(temp)
        main_one.append(test_row)

    
    for x in range(180):
        temp = []
        temp = np.hstack([main_one[0][x], main_one[1][x], main_one[2][x], main_one[3][x]])
        timeseries.append(temp)
    
    # Creating the correlation matrix: 
        
    # Dealing with nan:
    timeseriesnew = np.nan_to_num(timeseries)
    
    # Now for the correlation matrix: 
    d = dict(enumerate(timeseriesnew,1))
    columns = [i for i in range(361)]
    columns.remove(0)
    df = pd.DataFrame(d)
    
    corrMatrix = df.corr()
    
    # Comment out if needed
    print(corrMatrix)
    
    corr = corrMatrix.to_numpy()
    
    # Deal with the NaN's: 
    corr = np.nan_to_num(corr)
    
    # Now we need to extract the upper right triangle as the matrix is symmetric
    upper_triangle_flattened = corr[np.triu_indices(360,1)]
    
    all_X.append(upper_triangle_flattened)

# Saving all_X to be used as the final feature matrix
np.save('ts_X.npy', all_X)

# Print out completion message: 
print('Timeseries analysis completed. Feature matrix created and saved.')


