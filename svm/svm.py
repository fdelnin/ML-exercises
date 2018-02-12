from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.datasets import load_wine
wine=load_wine()

targetwine=wine.target
datawine=wine.data[:, [0, 11]] #0 is alcol 11 is hue(colore)

from sklearn.model_selection import train_test_split as tts
#split dei dati
datatrain,  datatest, targettrain, targettest= tts(datawine,targetwine, test_size=0.3, random_state=5)

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

def evaluate(y_test, y_pred):
	print "accurancy " , accuracy_score(y_test, y_pred)
	print "precision" , precision_score(y_test, y_pred, average=None)
	print "recall   " , recall_score(y_test, y_pred, average=None)
	print "f1 score ", f1_score(y_test,y_pred,average=None)

#polynomial kernel ha grado 3 di default
kernels=['linear','rbf','poly']
i=0
#salvo i valori max e min per disegnare il grafico
xmin = np.min(datawine[:, 0])-0.5
xmax = np.max(datawine[:, 0])+0.5
ymin = np.min(datawine[:, 1])-0.5
ymax = np.max(datawine[:, 1])+0.5

gs = gridspec.GridSpec(1, 3)
fig = plt.figure(figsize=(12, 4))

for kernel in kernels:
	clfsvm=svm.SVC(kernel=kernel)
	clfsvm.fit(datatrain,targettrain)

	y_preddd=clfsvm.predict(datatest)
	print ('\n kernel = ' + kernel)

	evaluate(targettest,y_preddd)

	ax = plt.subplot(gs[i])
	fig=plot_decision_regions(X=datawine, y=targetwine, clf=clfsvm, X_highlight=datatest, legend=0)

	ax.set_title(kernel)
	#gca=get current axes
	axes = plt.gca()
	#imposto il range degli assi per visualizzare meglio
	axes.set_xlim([xmin,xmax])
	axes.set_ylim([ymin,ymax])
	ax.set_ylabel('Hue')
	ax.set_xlabel('Alcohol')
	i=i+1

plt.suptitle('Wine dataset')
plt.savefig("svmWine.png")
plt.show()
