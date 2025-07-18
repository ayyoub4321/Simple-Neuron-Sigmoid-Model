import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class NeuronArtificiel:
    def __init__(self):
        self.W = None
        self.b = None

    def _initialisation(self, X):
        self.W = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn(1)
    def _model(self,X):
        Z=X.dot(self.W)+self.b
        Z = np.clip(Z, 1e-15, 1 - 1e-15)
        A=1/(1+np.exp(-Z))
        return A
    def _log_loss(self,A,y):
        A = np.clip(A, 1e-15, 1 - 1e-15)
        return 1/len(y) * np.sum(-y*np.log(A)-(1-y)*np.log(1-A))
    def _gradient(self,A,X,y):
        dw=1/len(y) * np.dot(X.T,A-y)
        db=1/len(y) * np.sum(A-y)
        return dw,db
    def _update(self,dw,db,learnin_rate):
        self.W=self.W-learnin_rate*dw
        self.b=self.b-learnin_rate*db
    def _predict(self,X):
        A=self._model(X)
        
        return A>=0.5
    def fit(self,X,y,learning_rate=0.1,n_iter=100):
        self._initialisation(X)
        Loss=[]
        for i in range(n_iter):
            A=self._model(X)
            loss=self._log_loss(A,y)
            Loss.append(loss)
            dw,db=self._gradient(A,X,y)
            self._update(dw,db,learning_rate)
        acc=accuracy_score(self._predict(X),y)
        print('ACC',acc)
        fig,axs=plt.subplots()
        axs.plot(Loss)
        plt.show()
        print('loss ',Loss)
        




