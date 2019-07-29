import Examine 
import CreateSamples

        #create PSD samples - deep anesthesia
sDAdelta, sDAtheta, sDAalpha, sDAbeta, sDAlowgamma = CreateSamples.PSDsamples('cl_data_DA', 2, 500, 7, '.txt')


        #create PSD samples - light anesthesia

#.mat files to .npy files & save
files= Examine.loaddmat('data_LA',7)   #load mat files
Examine.saveNpyfromMat_LA(files,'data_LA')  #save as .npy files
files= Examine.loaddmat('data_LAw',7) 
Examine.saveNpyfromMat_LAw(files,'data_LA')
files=Examine.loaddmat('Ldata',7)
Examine.saveNpyfromMat_L(files, 'Ldata') 

LA= Examine.loaddnpy('data_LA',7)     #load all .npy files 
clean_LA= CreateSamples.del_missingch_LA(LA)   #delete channels 33 and 44 for all subjects but nb2
Examine.saveclean(clean_LA, 7, 'clean_data_LA')

LAw= Examine.loaddnpy('data_LAw',7) 
clean_LAw= CreateSamples.del_missingch_LA(LAw)
Examine.saveclean(clean_LAw, 7, 'clean_data_LAw')


sLAdelta, sLAtheta, sLAalpha, sLAbeta, sLAlowgamma = CreateSamples.PSDsamples('cl_data_LA', 2, 500, 7, '.npy') #create samples per power band
sLAwdelta, sLAwtheta, sLAwalpha, sLAwbeta, sLAwlowgamma = CreateSamples.PSDsamples('cl_data_LAw', 2, 500, 7, '.npy')

Examine.save3Darray('sLAdelta.npy', sLAdelta)   #saving sample arrays                                                                                                                  
Examine.save3Darray('sLAtheta.npy', sLAtheta)                                                                                                                     
Examine.save3Darray('sLAalpha.npy', sLAalpha)                                                                                                                     
Examine.save3Darray('sLAbeta.npy', sLAbeta)                                                                                                                      
Examine.save3Darray('sLAlowgamma.npy', sLAlowgamma)  

Examine.save3Darray('sLAwdelta.npy', sLAwdelta)   #saving sample arrays                                                                                                                  
Examine.save3Darray('sLAwtheta.npy', sLAwtheta)                                                                                                                     
Examine.save3Darray('sLAwalpha.npy', sLAwalpha)                                                                                                                     
Examine.save3Darray('sLAwbeta.npy', sLAwbeta)                                                                                                                      
Examine.save3Darray('sLAwlowgamma.npy', sLAwlowgamma) 




        #create Topoplots - light anesthesia
avLAdelta, avLAtheta, avLAalpha, avLAbeta, avLAlowgamma= TopoPB.avPowerband('cl_data_LA',7, '.npy')
avLAwdelta, avLAwtheta, avLAwalpha, avLAwbeta, avLAwlowgamma= TopoPB.avPowerband('cl_data_LA',7, '.npy')
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

diffDLA = avDLA - avDLAw
diffTLA = avTLA - avTLAw
diffALA = avALA - avALAw
diffBLA = avBLA - avBLAw
diffLGLA = avLGLA - avLGLAw

        #plot topomaps
TopoPB.plotTopo(avDLA, avTLAw, avALAw, avBLAw, avLGLAw, avDLA, avTLA, avALA, avBLA, avLGLA, diffDLA, diffTLA, diffALA, diffBLA, diffLGLA, 'LA')


        #interpolate new electrode positions LA - saved 
elect_DA_new= TopoPB.interp_elect_pos('elect_LA','elect_DA')





'''
#code Thomas pour pas mm nombre d'éléctrodes

assert con.shape[0]==uncon.shape[0],    "attention, con et uncon pas le même nb d'elec"
    if con.shape[0] == 65:
        elec = np.ones(shape = (con.shape[0]),dtype = bool)
        elec[[39,44]] = False
        con = con[elec,:]
        uncon = uncon[elec,:]
    con = np.mean(con, axis =1)[...,np.newaxis]

'''