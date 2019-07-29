import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
import Examine 

def class_PSD_1class(classifier, files):

    file_namesDA=['sDAdelta.npy', 'sDAtheta.npy', 'sDAalpha.npy', 'sDAsigma.npy','sDAbeta.npy','sDAlowgamma.npy']
    file_namesDAw=['sDAwdelta.npy', 'sDAwtheta.npy', 'sDAwalpha.npy', 'sDAwsigma.npy','sDAwbeta.npy','sDAwlowgamma.npy']

    file_namesLA=['sLAdelta.npy', 'sLAtheta.npy', 'sLAalpha.npy', 'sLAsigma.npy','sLAbeta.npy','sLAlowgamma.npy']
    file_names=LAw=['sLAwdelta.npy', 'sLAwtheta.npy', 'sLAwalpha.npy', 'sLAwsigma.npy','sLAwbeta.npy','sLAwlowgamma.npy']

    listDA=[]
    listDAw=[]

    if files =='DA':
        file_names=file_namesDA
        file_names2=file_namesDAw
    elif files =='LA':
        file_names=file_namesLA
        file_names2=file_namesLAw


    for i, j in zip(file_names, file_names2):
        listDA.append(np.load(i))
        listDAw.append(np.load(j))

    listDA=np.concatenate(listDA, axis=2)     
    listDAw=np.concatenate(listDAw, axis=2)

    listDA=listDA.reshape((-1,listDA.shape[2]))   
    listDAw=listDAw.reshape((-1,listDAw.shape[2]))

    X=np.concatenate((listDA,listDAw), axis=0)
    y=np.concatenate((np.zeros(len(listDA)),np.ones(len(listDAw)))) 

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    if classifier == 'SVM':
        clf= svm.SVC(gamma='auto')
        clf.fit(X_train, y_train)

        print(clf.score(X_train, y_train))
        print(clf.score(X_test, y_test))

    if classifier == 'randomforest': 
        forest= RandomForestClassifier(criterion='entropy', n_estimators=10)
        forest.fit(X_train, y_train)

        print(forest.score(X_train, y_train))
        print(forest.score(X_test, y_test))

    if classifier == 'k_nearest':
        knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
        knn.fit(X_train, y_train)

        print(knn.score(X_train, y_train))
        print(knn.score(X_test, y_test))

    if classifier == 'LDA':
        lda= LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)

        #LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)
        #print(clf.predict([[-0.8, -1]]))
        #???

        print(lda.score(X_train, y_train))
        print(lda.score(X_test, y_test))

    if classifier == 'QDA':

        qda=QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)

        print(lda.score(X_train, y_train))
        print(lda.score(X_test, y_test))

    if classifier == 'MLP':
        mlp=MLPClassifier()
        mlp.fit(X_train, y_train)

        print(mlp.score(X_train, y_train))
        print(mlp.score(X_test, y_test))


def class_PSD(files):

    file_namesDA=['sDAdelta.npy', 'sDAtheta.npy', 'sDAalpha.npy','sDAbeta.npy','sDAlowgamma.npy']
    file_namesDAw=['sDAwdelta.npy', 'sDAwtheta.npy', 'sDAwalpha.npy','sDAwbeta.npy','sDAwlowgamma.npy']

    file_namesLA=['sLAdelta.npy', 'sLAtheta.npy', 'sLAalpha.npy','sLAbeta.npy','sLAlowgamma.npy']
    file_namesLAw=['sLAwdelta.npy', 'sLAwtheta.npy', 'sLAwalpha.npy','sLAwbeta.npy','sLAwlowgamma.npy']

    listDA=[]
    listDAw=[]



    if files =='DA':
        file_names=file_namesDA
        file_names2=file_namesDAw

    elif files =='LA':
        file_names=file_namesLA
        file_names2=file_namesLAw


    for i, j in zip(file_names, file_names2):
        listDA.append(np.load(i, allow_pickle=True))
        listDAw.append(np.load(j, allow_pickle=True))

    listDA=np.concatenate(listDA, axis=2)     
    listDAw=np.concatenate(listDAw, axis=2)

    listDA=listDA.reshape((-1,listDA.shape[2]))   
    listDAw=listDAw.reshape((-1,listDAw.shape[2]))

    X=np.concatenate((listDA,listDAw), axis=0)
    y=np.concatenate((np.zeros(len(listDA)),np.ones(len(listDAw)))) 

    for i in range(X.shape[1]):
        X_train, X_test, y_train, y_test = train_test_split(X[:,i],y)

        clf= svm.SVC(gamma='auto')
        clf.fit(X_train, y_train)

        print('SVM: ' , clf.score(X_train, y_train))
        print('SVM: ', clf.score(X_test, y_test))

 
    forest= RandomForestClassifier(criterion='entropy', n_estimators=10)
    forest.fit(X_train, y_train)

    print('Random Forest: ', forest.score(X_train, y_train))
    print('Random Forest: ', forest.score(X_test, y_test))

    
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(X_train, y_train)

    print('K nearest neighbor: ', knn.score(X_train, y_train))
    print('K nearest neighbor: ', knn.score(X_test, y_test))


    lda= LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

        #LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)
        #print(clf.predict([[-0.8, -1]]))
        #???

    print('LDA: ', lda.score(X_train, y_train))
    print('LDA: ',lda.score(X_test, y_test))


    qda=QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)

    print('LDA: ', lda.score(X_train, y_train))
    print('LDA: ',lda.score(X_test, y_test))


    mlp=MLPClassifier()
    mlp.fit(X_train, y_train)

    print('Multi-layer perceptron: ', mlp.score(X_train, y_train))
    print('Multi-layer perceptron: ',mlp.score(X_test, y_test))


'''
results:

SVM:  0.5684389140271493
SVM:  0.5847457627118644

Random Forest:  0.5684389140271493
Random Forest:  0.5847457627118644

K nearest neighbor:  0.9949095022624435
K nearest neighbor:  0.9830508474576272

/Users/alina/anaconda3/envs/mne/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")

LDA:  0.9994343891402715
LDA:  0.976271186440678

/Users/alina/anaconda3/envs/mne/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:692: UserWarning: Variables are collinear
  warnings.warn("Variables are collinear")

LDA:  0.9994343891402715
LDA:  0.976271186440678

Multi-layer perceptron:  0.5684389140271493
Multi-layer perceptron:  0.5847457627118644

'''



# class signe feature -- signgle electrode 

def class_sf(files):
  
    liste=[0,1,2,3,4,5]

    accu_clf = []
    accu_forest=[]
    accu_knn=[]
    accu_lda=[]
    accu_qda=[]
    accu_mlp=[]

    file_namesDA=['sDAdelta.npy', 'sDAtheta.npy', 'sDAalpha.npy','sDAbeta.npy','sDAlowgamma.npy']
    file_namesDAw=['sDAwdelta.npy', 'sDAwtheta.npy', 'sDAwalpha.npy','sDAwbeta.npy','sDAwlowgamma.npy']

    file_namesLA=['sLAdelta.npy', 'sLAtheta.npy', 'sLAalpha.npy','sLAbeta.npy','sLAlowgamma.npy']
    file_namesLAw=['sLAwdelta.npy', 'sLAwtheta.npy', 'sLAwalpha.npy','sLAwbeta.npy','sLAwlowgamma.npy']

    listDA=[]
    listDAw=[]

    if files =='DA':
        file_names=file_namesDA
        file_names2=file_namesDAw

    elif files =='LA':
        file_names=file_namesLA
        file_names2=file_namesLAw

    for i, j in zip(file_names, file_names2):

        listDA.append(np.load(i, allow_pickle=True))
        listDAw.append(np.load(j, allow_pickle=True))

    listDA=np.concatenate(listDA, axis=2)     
    listDAw=np.concatenate(listDAw, axis=2)

    listDA=listDA.reshape((-1,listDA.shape[2]))   
    listDAw=listDAw.reshape((-1,listDAw.shape[2]))

    X=np.concatenate((listDA,listDAw), axis=0)
    y=np.concatenate((np.zeros(listDA.shape[0]),np.ones(listDAw.shape[0])))

    X=X.T

    for i in range(X.shape[0]):
        
        x = X[i,:]
        x = x.reshape((-1,1))

        X_train, X_test, y_train, y_test = train_test_split(x,y)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        clf= svm.SVC(gamma='auto')
        clf.fit(X_train, y_train)

        forest= RandomForestClassifier(criterion='entropy', n_estimators=10)
        forest.fit(X_train, y_train)

        knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
        knn.fit(X_train, y_train)

        lda= LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)

        qda=QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)

        mlp=MLPClassifier()
        mlp.fit(X_train, y_train)


        accu_clf.append(clf.score(X_test,y_test))
        accu_forest.append(forest.score(X_test,y_test))
        accu_knn.append(knn.score(X_test,y_test))
        accu_lda.append(lda.score(X_test,y_test))
        accu_qda.append(qda.score(X_test,y_test))
        accu_mlp.append(mlp.score(X_test,y_test))

    return accu_clf, accu_forest, accu_knn, accu_lda, accu_qda, accu_mlp

def mean_score():

    mean_sc=np.zeros((63,6))
    divider = np.full((63,6), 6)

    for i,j in zip(range(accu_clf.shape[0]), range(accu_clf.shape[1])):
        mean_sc[i,j]= accu_clf[i,j] + accu_forest[i,j] + accu_knn[i,j] + accu_lda[i,j] + accu_qda[i,j] + accu_mlp

    mean_sc/=divider 

    return mean_sc

def split_class(accu_clf, accu_forest, accu_knn, accu_lda, accu_qda, accu_mlp, nb_electrodes):

    clf_delta=[accu_clf[:nb_electrodes]]
    clf_theta=[accu_clf[nb_electrodes:nb_electrodes*2]]
    clf_alpha=[accu_clf[nb_electrodes*2:nb_electrodes*3]]
    clf_beta=[accu_clf[nb_electrodes*3:nb_electrodes*4]]
    clf_lowgamma=[accu_clf[nb_electrodes*4:nb_electrodes*5]]

    forest_delta=[accu_forest[:nb_electrodes]]
    forest_theta=[accu_forest[nb_electrodes:nb_electrodes*2]]
    forest_alpha=[accu_forest[nb_electrodes*2:nb_electrodes*3]]
    forest_beta=[accu_forest[nb_electrodes*3:nb_electrodes*4]]
    forest_lowgamma=[accu_forest[nb_electrodes*4:nb_electrodes*5]]

    knn_delta=[accu_knn[:nb_electrodes]]
    knn_theta=[accu_knn[nb_electrodes:nb_electrodes*2]]
    knn_alpha=[accu_knn[nb_electrodes*2:nb_electrodes*3]]
    knn_beta=[accu_knn[nb_electrodes*3:nb_electrodes*4]]
    knn_lowgamma=[accu_knn[nb_electrodes*4:nb_electrodes*5]]
    
    lda_delta=[accu_lda[:,nb_electrodes]]
    lda_theta=[accu_lda[nb_electrodes:nb_electrodes*2]]
    lda_alpha=[accu_lda[nb_electrodes*2:nb_electrodes*3]]
    lda_beta=[accu_lda[nb_electrodes*3:nb_electrodes*4]]
    lda_lowgamma=[accu_lda[nb_electrodes*4:nb_electrodes*5]]

    qda_delta=[accu_qda[:,nb_electrodes]]
    qda_theta=[accu_qda[nb_electrodes:nb_electrodes*2]]
    qda_alpha=[accu_qda[nb_electrodes*2:nb_electrodes*3]]
    qda_beta=[accu_qda[nb_electrodes*3:nb_electrodes*4]]
    qda_lowgamma=[accu_qda[nb_electrodes*4:nb_electrodes*5]]

    mlp_delta=[accu_mlp[:,nb_electrodes]]
    mlp_theta=[accu_mlp[nb_electrodes:nb_electrodes*2]]
    mlp_alpha=[accu_mlp[nb_electrodes*2:nb_electrodes*3]]
    mlp_beta=[accu_mlp[nb_electrodes*3:nb_electrodes*4]]
    mlp_lowgamma=[accu_mlp[nb_electrodes*4:nb_electrodes*5]]


    list_clf=Examine.appendd(clf_delta, clf_theta, clf_alpha, clf_beta, clf_lowgamma)
    list_forest=Examine.appendd(forest_delta, forest_theta, forest_alpha, forest_beta, forest_lowgamma)
    list_knn=Examine.appendd(knn_delta, knn_theta, knn_alpha, knn_beta, knn_lowgamma)
    list_lda=Examine.appendd(lda_delta, lda_theta, lda_alpha, lda_beta, lda_lowgamma)
    list_qda=Examine.appendd(qda_delta, qda_theta, qda_alpha, qda_beta, qda_lowgamma)
    list_mlp=Examine.appendd(mlp_delta, mlp_theta, mlp_alpha, mlp_beta, mlp_lowgamma)


    return list_clf, list_forest, list_knn, list_lda, list_qda, list_mlp


    







     


















    

