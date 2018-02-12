import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
digits=load_digits()
X=digits.data
y=digits.target

from sklearn.model_selection import train_test_split as tts
#split dei dati
Xtrain,  Xtest, ytrain, ytest= tts(X,y, test_size=0.3,random_state=5)

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as nn
from sklearn.metrics import accuracy_score

bestsvm=0
bestnn=0
j=0
nhidden=0
accsvm=[]
accsnn=[]

#gradi da 2 a 12 / neuroni nel layer nascosto da 20 a 120
for i in range(2,13):

	print "degree on polynomial kernel/number of neuron in the hidden layer(*10) = "+str(i)
	#svm
	Svm=SVC(C=1,kernel='poly', degree=i)
	Svm.fit(Xtrain,ytrain)
	ypred=Svm.predict(Xtest)
	acc= accuracy_score(ytest, ypred)
	
	accsvm.append(acc)
	
	print "accuracy on svm= "+str(acc)
	#salvo il valore migliore
	if(acc>bestsvm):
		bestsvm=acc
		j=i

	#neural network con un hodden layer, varia il numero di perceptron 20 a 120
	NN=nn(hidden_layer_sizes=(i*10),random_state=5,learning_rate_init=0.001,solver='sgd',max_iter=300)
	NN.fit(Xtrain,ytrain)
	yprednn=NN.predict(Xtest)
	accnn= accuracy_score(ytest, yprednn)

	accsnn.append(accnn)
	
	print "accuracy on nn = "+str(accnn)
	print ''
	#salvo il valore migliore
	if(accnn>bestnn):
		bestnn=accnn
		nhidden=i*10


print "Best accuracy on svm at degree = "+str(j)

print "Best accuracy on nerual netrwork at "+str(nhidden)+" perceptron in hidden layer"

#plot
plt.title("Digit dataset")
plt.xlabel("degree on poly kernel/perceptron in hidden layer (*10)")
plt.ylabel("accuracy")
plt.plot(np.arange(2, 13) , accsvm,'r-',label='svm polynomial kernel')
plt.plot(np.arange(2, 13), accsnn,'b-',label='neural network')

plt.legend()
plt.savefig('accuracy.png')
plt.show()