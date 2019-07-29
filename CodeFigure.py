import TopoPB 
import numpy as np 
import CreateSamples
import scipy.interpolate as interp
import Class2


#topoplot average Power bands : Deep Anesthesia
'''
avDDA, avTDA, avADA, avBDA, avLGDA= TopoPB.avPowerband('cl_data_DA',9)
avDDAw, avTDAw, avADAw, avBDAw, avLGDAw= TopoPB.avPowerband('cl_data_DAw',9)
'''

avDDA=np.loadtxt('avDelta_DA.txt')
avTDA=np.loadtxt('avTheta_DA.txt')
avADA=np.loadtxt('avAlpha_DA.txt')
avBDA=np.loadtxt('avBeta_DA.txt')
avLGDA=np.loadtxt('avLowgamma_DA.txt')

avDDAw=np.loadtxt('avDelta_DAw.txt')
avTDAw=np.loadtxt('avTheta_DAw.txt')
avADAw=np.loadtxt('avAlpha_DAw.txt')
avBDAw=np.loadtxt('avBeta_DAw.txt')
avLGDAw=np.loadtxt('avLowgamma_DAw.txt')



avDLA=np.load('avLAdelta.npy')
avTLA=np.load('avLAtheta.npy')
avALA=np.load('avLAalpha.npy')
avBLA=np.load('avLAbeta.npy')
avLGLA=np.load('avLAlowgamma.npy')

avDLAw=np.load('avLAwdelta.npy')
avTLAw=np.load('avLAwtheta.npy')
avALAw=np.load('avLAwalpha.npy')
avBLAw=np.load('avLAwbeta.npy')
avLGLAw=np.load('avLAwlowgamma.npy')



#saved 
avDLA=np.load('avLAdelta.npy')
avTLA=np.load('avLAtheta.npy')
avALA=np.load('avLAalpha.npy')
avBLA=np.load('avLAbeta.npy')
avLGLA=np.load('avLAlowgamma.npy')

avDLAw=np.load('avLAwdelta.npy')
avTLAw=np.load('avLAwtheta.npy')
avALAw=np.load('avLAwalpha.npy')
avBLAw=np.load('avLAwbeta.npy')
avLGLAw=np.load('avLAwlowgamma.npy')


interp_avDDA= np.load('interp_avDDA.npy')
interp_avTDA= np.load('interp_avTDA.npy')
interp_avADA= np.load('interp_avADA.npy')
interp_avBDA= np.load('interp_avBDA.npy')
interp_avLGDA= np.load('interp_avLGDA.npy')

list_DA=Examine.appendd(avDDA, avTDA, avADA, avBDA, avLGDA)
list_DAw= Examine.appendd(avDDAw, avTDAw, avADAw, avBDAw, avLGDAw)
list_DA_interp= Examine.appendd(interp_avDDA, interp_avTDA, interp_avADA, interp_avBDA, interp_avLGDA)

list_LA=Examine.appendd(avALA, avTLA, avALA, avBLA, avLGLA)
list_LAw= Examine.appendd(avDLAw, avTLAw, avALAw, avBLAw, avLGLAw)

diffD = avDDA - avDDAw
diffT = avTDA - avTDAw
diffA = avADA - avADAw
diffB = avBDA - avBDAw
diffLG = avLGDA - avLGDAw

diffDLA = avDLA - avDLAw   #L2 norm euclidische 
diffTLA = avTLA - avTLAw
diffALA = avALA - avALAw
diffBLA = avBLA - avBLAw
diffLGLA = avLGLA - avLGLAw


accu_clf=np.load('accu_clf_DA.npy')
accu_forest=np.load('accu_forest_DA.npy')                                                                                                                        
accu_knn=np.load('accu_knn_DA.npy')                                                                                                                              
accu_lda=np.load('accu_lda_DA.npy')                                                                                                                              
accu_qda=np.load('accu_qda_DA.npy')                                                                                                                              
accu_mlp=np.load('accu_mlp_DA.npy')     

list_clf, list_forest, list_knn, list_lda, list_qda, list_mlp =Class2.split_class(accu_clf, accu_forest, accu_knn, accu_lda, accu_qda, accu_mlp, 85)                                                                                                                                                   


TopoPB.plotTopo(avDDA, avTDAw, avADAw, avBDAw, avLGDAw, avDDA, avTDA, avADA, avBDA, avLGDA, diffD, diffT, diffA, diffB, diffLG)

idx2rem=[] 
ch2rem= CreateSamples.find_missingch('data_DA','data_DAw', 9 ) 

montage=mne.channels.read_montage('GSN-HydroCel-128') 

for k in range(len(ch2rem)): 
    if ch2rem[k] in montage.ch_names: 
        idx2rem.append(montage.ch_names.index(ch2rem[k]))

idx2rem.extend((118,120,126))    #too  high values

montage.pos=np.delete(montage.pos, idx2rem, axis=0)
montage.ch_names=np.delete(montage.ch_names, idx2rem, axis=0)

sensors_pos=montage.get_pos2d()[3:]
ch_names_info=montage.ch_names[3:]


        #create Topoplots - light anesthesia
avLAdelta, avLAtheta, avLAalpha, avLAbeta, avLAlowgamma= TopoPB.avPowerband('cl_data_LA',7, '.npy')
avLAwdelta, avLAwtheta, avLAwalpha, avLAwbeta, avLAwlowgamma= TopoPB.avPowerband('cl_data_LA',7, '.npy')





        #plot topomaps
TopoPB.plotTopo(avDLA, avTLAw, avALAw, avBLAw, avLGLAw, avDLA, avTLA, avALA, avBLA, avLGLA, diffDLA, diffTLA, diffALA, diffBLA, diffLGLA, 'LA')


        #interpolate new electrode positions LA - saved 
elect_DA_new= TopoPB.interp_elect_pos('elect_LA','elect_DA')  #saved





interp_data_DA= TopoPB.interp_data(list_DA, 'elect_DA', 'elect_LA', 'elect_DA_new')

TopoPB.plotTopo(list_LA, interp_data_DA, 'light', 'deep', 'mixed')



'''
np.savetxt('avDelta_DAw.txt', avDDAw)
np.savetxt('avTheta_DAw.txt', avTDAw)
np.savetxt('avAlpha_DAw.txt', avADAw)
np.savetxt('avBeta_DAw.txt', avBDAw)
np.savetxt('avLowgamma_DAw.txt', avLGDAw)
'''


fig= plt.figure()

    ax1=plt.subplot(3,5,1)
    ax1.mne.viz.plot_topomap(avDeltaDAw,sensors_pos,show=False, cmap='coolwarm')
    plt.title('Delta')  #2-4 Hz

    ax2=plt.subplot(3,5,2)
    ax2.mne.viz.plot_topomap(avThetaDAw,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Theta')  #5-7 Hz

    ax3=plt.subplot(3,5,3)
    ax3.mne.viz.plot_topomap(avAlphaDAw,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Alpha')  #8-13 Hz




    ax4=plt.subplot(3,5,4)
    ax4.mne.viz.plot_topomap(avBetaDAw,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Beta')   #13-30

    ax5=plt.subplot(3,5,5)
    ax5.mne.viz.plot_topomap(avLowgammaDAw,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Low Gamma')  #30-50


    ax6=plt.subplot(3,5,6)
    ax6.mne.viz.plot_topomap(avDeltaDA,sensors_pos,show=False, cmap='coolwarm')
    plt.title('Delta')  #2-4 Hz

    ax7= plt.subplot(3,5,7)
    ax7.mne.viz.plot_topomap(avThetaDA,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Theta')  #5-7 Hz

    ax8=plt.subplot(3,5,8)
    ax8.mne.viz.plot_topomap(avAlphaDA,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Alpha')  #8-13 Hz

    ax9=plt.subplot(3,5,9)
    ax9.mne.viz.plot_topomap(avBetaDA,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Beta')   #13-30

    ax10=plt.subplot(3,5,10)
    ax10.mne.viz.plot_topomap(avLowgammaDA,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Low Gamma')  #30-50


    ax11=plt.subplot(3,5,11)
    ax11.mne.viz.plot_topomap(DiffD,sensors_pos,show=False, cmap='coolwarm')
    plt.title('Delta')  #2-4 Hz

    ax12=plt.subplot(3,5,12)
    ax12.mne.viz.plot_topomap(DiffT,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Theta')  #5-7 Hz

    ax13=plt.subplot(3,5,13)
    ax13.mne.viz.plot_topomap(DiffA,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Alpha')  #8-13 Hz

    ax14=plt.subplot(3,5,14)
    ax14.mne.viz.plot_topomap(DiffB,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Beta')   #13-30

    ax15=plt.subplot(3,5,15)
    ax15.mne.viz.plot_topomap(DiffLG,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Low Gamma')  #30-50

    ax1.text(0.1,0.9, 'wakefulness')
    ax6.text(0.4,0.9, 'deep anesthesia')
    ax11.text(0.8,0.9, 'wake - deep')

