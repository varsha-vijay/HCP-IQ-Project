# A script to visualise the mean features and check the hand engineered features are correct: 
    
# First importing the relevant libraries:
# Import libraries: 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel
import time
import copy

# Choose one subject, and calculate the mean of each feature at each region: 
featuresL = nibabel.load('C:\\Users\\priya\\Documents\\MSc Extended Research Project\\allfeatures\\100307.L.MultiModal_Features_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.func.gii')
featuresR = nibabel.load('C:\\Users\\priya\\Documents\\MSc Extended Research Project\\allfeatures\\100307.R.MultiModal_Features_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.func.gii')

featuresLMean = copy.deepcopy(featuresL)
featuresRMean = copy.deepcopy(featuresR)

labelL = nibabel.load('C:\\Users\\priya\\Documents\\MSc Extended Research Project\\alllabels\\100307.L.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')
labelR = nibabel.load('C:\\Users\\priya\\Documents\\MSc Extended Research Project\\alllabels\\100307.R.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')

label_names_L = labelL.labeltable.get_labels_as_dict()
label_names_R = labelR.labeltable.get_labels_as_dict()
np.array([0])

if (len(label_names_L)==181):
    row = []
    
    for i in np.arange(1,len(label_names_L)):
        for j in range(featuresL.numDA):
            # Variable to give location of each label.
            label_pos_L = np.where(labelL.darrays[0].data == i )[0]
            
            temp_mean_L = np.mean(featuresL.darrays[j].data[label_pos_L])
            featuresLMean.darrays[j].data[label_pos_L] = temp_mean_L

            label_pos_R = np.where(labelR.darrays[0].data == i )[0]
            
            temp_mean_R = np.mean(featuresR.darrays[j].data[label_pos_R])
            featuresRMean.darrays[j].data[label_pos_R] = temp_mean_R

            # FOR MEAN
            # row.append(temp_mean_L)
            # row.append(temp_mean_R)
            

# Now saving the files:
nibabel.save(featuresLMean, 'Ltest_features_mean.func.gii')
nibabel.save(featuresRMean, 'Rtest_features_mean.func.gii')
            
