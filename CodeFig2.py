import TopoPB 
import numpy as np 
import CreateSamples
import scipy.interpolate as interp
import Class2
import Examine 


scaled_DA= np.load('scaled_list_DA.npy')
scaled_DAw= np.load('scaled_list_DAw.npy')
scaled_LA= np.load('scaled_list_LA.npy')
scaled_LAw= np.load('scaled_list_LAw.npy')
scaled_interp_DA= np.load('list_scaled_interp_DA.npy')

scaled_diff_DA=[]
scaled_diff_LA=[]
scaled_diff_interp=[]

for i in range(len(scaled_DA)):
    scaled_diff_DA.append(scaled_DA[i]-scaled_DAw[i])
    scaled_diff_LA.append(scaled_LA[i]-scaled_LAw[i])
    scaled_diff_interp.append(scaled_interp_DA[i]-scaled_LA[i])


list_clfLAT= np.load('list_clf_LAT.npy')
list_forestLAT= np.load('list_forest_LAT.npy')
list_knnLAT= np.load('list_knn_LAT.npy')
list_ldaLAT = np.load('list_lda_LAT.npy')
list_qdaLAT = np.load('list_qda_LAT.npy')
list_mlpLAT = np.load('list_mlp_LAT.npy')

mean_sc_LA= Class2.mean_score(list_clfLAT, list_forestLAT, list_knnLAT, list_ldaLAT, list_qdaLAT, list_mlpLAT)  

sensors_pos_DA= np.load('elect_pos_DA_clean.npy')
sensors_pos_LA = np.load('elect_pos_LA.npy')
#sensors_pos_DA_interp= np.load('elect_DA_new.npy')