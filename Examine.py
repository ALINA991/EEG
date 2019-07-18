import mne 
import numpy as np
import scipy.io

def printShape(file_name, nb_subjects, ext):

    files=[]
    subjects=range(nb_subjects)

    if ext == '.edf':  
        for i in subjects:
            files.append(file_name +str(i)+ '.edf')  
        for i in subjects: 
            raw= mne.io.read_raw_edf(files[i])    #load data from edf file
            data=raw.get_data()
            print(i, data.shape)

    if ext == '.npy':
        for i in subjects:
            files.append(file_name +str(i)+ '.npy')

        for i in subjects:                      #load data from npy file
            data=np.load(files[i])
            print(i, data.shape)

    if ext == '.txt':
        for i in subjects:
             files.append(file_name +str(i)+ '.txt')

        for i in subjects:                      #load data from text file
            data=np.loadtxt(files[i])
            print(i, data.shape)

    if ext == 'none':
        data=file_name
        for i in subjects:
            print(i, data[i].shape)
        


def getSmallestItemNb(file_name, nb_subjects, ext): 

    subjects=range(nb_subjects)
    lens=[]
 
    if ext == '.edf':
        for i in subjects: 
                raw=mne.io.read_raw_edf(file_name +str(i) + ext)
                data=raw.get_data()
                lens.append(np.size(data,1))

    elif ext == '.npy':
        for i in subjects: 
                data= np.load(file_name + str(i) + ext)
                lens.append(np.size(data,1))

    elif ext == '.txt':
        for i in subjects: 
            data= np.loadtxt(file_name + str(i) + ext)
            lens.append(np.size(data,1))

    return min(lens)


def save3Darray(file_name, list):
    array= np.array(list)
    np.save(file_name,array)


def loaddmat(file_name, nb_subjects):

    files=[]
    for i in range(nb_subjects):
        mat = scipy.io.loadmat(file_name+str(i)+'.mat')
        files.append(mat)
    return files
    
def loaddnpy(file_name, nb_subjects):

    files=[]
    for i in range(nb_subjects):
        npy = np.load(file_name+str(i)+'.npy')
        files.append(npy)
    return files

def saveNpyfromMat_LA(files, new_file_name):
    dictio={0:'S2_unconscious_RawData', 1: 'S3_unconscious_RawData', 2: 'S4_unconscious_RawData', 3: 'S5_unconscious_RawData', 4: 'S6_unconscious_RawData', 5:'S8_unconscious_RawData', 6:'S10_unconscious_RawData'}
    
    for i in dictio: 
        data= files[i][dictio[i]]
        np.save(new_file_name +str(i)+'.npy',data)

def saveNpyfromMat_LAw(files, new_file_name ):

    #file n* 
    dictio={ 0:'S2_conscious_RawData', 1: 'S3_conscious_RawData', 2: 'S4_conscious_RawData', 3: 'S5_conscious_RawData', 4: 'S6_conscious_RawData', 5:'S8_conscious_RawData', 6:'S10_conscious_RawData'}
    
    for i in dictio: 
        data= files[i][dictio[i]]
        np.save(new_file_name +str(i)+'.npy',data)

def saveNpyfromMat_L(files, new_file_name):

    #file n* 
    dictio={ 0: 'S2_RawData', 1: 'S3_RawData', 2: 'S4_RawData', 3: 'S5_RawData', 4: 'S6_RawData', 5:'S8_RawData', 6:'S10_RawData'}
    
    for i in dictio: 
        np.save(new_file_name +str(i)+'.npy',files[i][dictio[i]])

def saveclean(data_list, nb_subjects, new_file_name):

    for i in range(nb_subjects):
        np.save(new_file_name +str(i)+'.npy', data_list[i])
        
