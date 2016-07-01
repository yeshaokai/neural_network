import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from io import StringIO
import numpy as np
from MLP_Linear import *
from scipy import signal
#from MLP import *

#df_air = pd.read_csv('AirQualityUCI.csv',sep=';')
#O3 = df_air.iloc[:,11].values.astype(float)
#df_air.dropna(axis=0)
#ss = StandardScaler()
#O3 = O3.reshape(len(O3),1)

lags = 3

print "the distance between input and output is %s" % (lags)
n_sample = 300

t = np.linspace(0,1,500,endpoint=False)
sig = np.sin(2*np.pi*t)

#plt.plot(t,sig)
#plt.ylim(-2,2)
#plt.show()

print len(t)

start_index = 0

y = sig[start_index+lags:start_index+lags+n_sample]
X = sig[start_index:start_index+n_sample+lags]
X_container = np.zeros((lags,n_sample))
for i in range(n_sample):
    X_lags = (X[start_index+i:start_index+i+lags]).flatten()
    X_container[:,i] = X_lags

#print X_container.shape
#print y.shape
n_iter = 1000
eta = 0.002
inodes = lags
onodes = 1
hnodes = 2
learning_curve = True

mlp = MLP(n_iter=n_iter,inodes=inodes,hnodes=hnodes,onodes=onodes,eta=eta,learning_curve=True,minibatches=5,lamda1 = 0.1,check_gradient=True)
y = y.reshape(1,len(y))
mlp.fit(X_container,y)


#mlp.draw_learning_curve()
print np.max(mlp.predict(X_container))
print (abs(mlp.predict(X_container)-y).mean()/abs(y).mean())

