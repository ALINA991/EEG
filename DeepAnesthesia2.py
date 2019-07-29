import mne
import numpy as np
import FeaturesExtraction 
from mne.event import define_target_events
from mne.channels import make_1020_channel_selections
import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt



    #dictionnary to acces data files (1-8) corresponding to subject n*:
subjects={0:3, 1:5, 2:6, 3:7, 4:10, 5:11, 6:12, 7:15, 8:17}

#bands={0:'delta',1:'theta',2:'alpha',3:'sigma',4:'beta',5:'low gamma'}

    #empty (NaN)
powerBlist=[pd.DataFrame(columns=['delta','theta','alpha','sigma','beta','low gamma'],index=range(129)) for x in range(9)]


    #filled with zeros
ave=np.zeros((128,6))
divider=np.zeros((128,6))


   


for i in subjects:
    
    raw=mne.io.read_raw_edf(files[i])       #importing raw data 
    events=mne.events_from_annotations(raw)      #extracting events 
 

    for j in range(128):
        
        try:
            data=raw.get_data()[j]    #extract data for each channel
        except:
            break                     #if less then 129 channels

        PSD=FeaturesExtraction.computePSD(data,300)     #compute power spectral denstity for each channel
        f=PSD[0]                      #assign frequency and power spectral density to 2 different vectors
        p=PSD[1]


        powerBrow=FeaturesExtraction.computePowerBands(f, p)   #compute absolute power bands for each channel
        powerBlist[i].iloc[j]=powerBrow                    #core.frame.DataFrame containing absolute power bands for each channel
    
    
    for k in range(128):            #each channel
        for y in range(6):           #each band
            if not np.isnan(powerBlist[i].iloc[k,y]):       
                ave[k,y] += powerBlist[i].iloc[k,y]     #sum dataframes element by element
                divider[k,y] += 1     

ave/=divider                 #average power band per electrode for all subjects
#np.savetxt('average_PB.txt',ave)        #save file 

ave=np.delete(ave, (118,120,126),axis=0)        #delete electrode with too high values
   
av_delta=ave[:,0]           #extract average for each band
av_theta=ave[:,1]
av_alpha=ave[:,2]
av_sigma=ave[:,3]
av_beta=ave[:,4]
av_lowgamma=ave[:,5]


        #get electrode postition from standard 128 Electrical Geodesics EEG 
montage=mne.channels.read_montage('GSN-HydroCel-128') 
sensors_pos=montage.get_pos2d()[3:]
ch_names_info=montage.ch_names[3:]

sensors_pos=np.delete(sensors_pos, (118,120,126),axis=0)


        #plot topomaps
fig, axes = plt.subplots(1,6)
mne.viz.plot_topomap(av_delta,sensors_pos,axes=axes[0],show=False)
mne.viz.plot_topomap(av_theta,sensors_pos,axes=axes[1],show=False)
mne.viz.plot_topomap(av_alpha,sensors_pos,axes=axes[2],show=False)
mne.viz.plot_topomap(av_sigma,sensors_pos,axes=axes[3],show=False)
mne.viz.plot_topomap(av_beta,sensors_pos,axes=axes[4],show=False)
mne.viz.plot_topomap(av_lowgamma,sensors_pos,axes=axes[5],show=False)
plt.show(block=False)



