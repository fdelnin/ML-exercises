import numpy as np
from sklearn.cluster import FeatureAgglomeration as fa
from sklearn.datasets import load_wine

wine=load_wine()
X=wine.data
Y=wine.target

print ('Number of X columns:')
print X.shape[1]

#le righe(esempi) saranno lo stesso numero ma le colonne (feature) saranno ridotte
C=fa(n_clusters=12,linkage ="ward") #ward=minimizes the variance of the clusters being merged
N=C.fit_transform(X)
print ('Number of N columns:')
print N.shape[1]
#C.labels_ = cluster labels for each feature
print('\nCluster labels for each feature:')
print C.labels_
print('Original features names:')
print wine.feature_names 

#seconda parte
from sklearn import svm
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate(y_test, y_pred):
	print "accurancy " , accuracy_score(y_test, y_pred)
	print "precision" , precision_score(y_test, y_pred, average=None)
	print "recall   " , recall_score(y_test, y_pred, average=None)

kernel='linear'
#split dei dati per X
Xtrain,  Xtest, Ytrain, Ytest= tts(X, Y, test_size=0.3, random_state=5)

Svm2=svm.SVC(kernel=kernel)
Svm2.fit(Xtrain,Ytrain)

print ('\nsvm kernel '+kernel+' on original dataset:')
evaluate(Ytest,Svm2.predict(Xtest))

#split dei dati per N
Ntrain,  Ntest, Ytrain, Ytest= tts(N, Y, test_size=0.3, random_state=5)

Svm=svm.SVC(kernel=kernel)
Svm.fit(Ntrain,Ytrain)

print ('\nsvm kernel '+kernel+' after clustering:')
evaluate(Ytest,Svm.predict(Ntest))
