import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris=load_iris()

#utilizzo solo i primi due parametri e considero solo 'setosa' e 'versicolor'
targetiris=np.array(filter(lambda x : x != 2, iris.target))
l=targetiris.shape[0]
datairis=iris.data[:l, [0, 1]]

from sklearn.model_selection import train_test_split as tts
#split dei dati
datatrain,  datatest, targettrain, targettest= tts(datairis,targetiris, test_size=0.25, random_state=5)

from sklearn.linear_model import Perceptron
perceptron=Perceptron(max_iter=1000,eta0=0.2,random_state=5)

perceptron.fit(datatrain,targettrain)
b=perceptron.predict(datatest)
print('predicted:')
print b
print('actual target:')
print targettest
print 'accurancy on test set: '+ str(perceptron.score(datatest,targettest)*100)+' %'

w=perceptron.coef_[0]
k=perceptron.intercept_

def f(x):
    return -(k+w[0]*x)/w[1] 

#disegno i punti:
#esempi di training
dot=plt.scatter(datatrain[:,0],datatrain[:,1], c=targettrain,label='train')
#esempi di test (disegnati con un diamante)
diamond=plt.scatter(datatest[:,0],datatest[:,1], c=targettest, marker='D',label='test')
plt.legend(handles=[dot,diamond])
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.title('Iris dataset')

plt.plot([4,6,7],[f(4),f(6),f(7)], 'g-')
plt.savefig('iris.png')
plt.show()