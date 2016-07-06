import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from io import StringIO
import numpy as np
from RNN import *
from scipy import signal

'''
this is a simple demonstration of time-Lagged Network

'''

lags = 1

print "the distance between input and output is %s" % (lags)
n_all=1000
n_sample = 700

t = np.linspace(0,1,n_all,endpoint=False)
sig = np.sin(2*np.pi*t)
mu,sigma =0.0,0.01
s = np.random.normal(mu,sigma,n_all)
sig = sig+s
#plt.plot(t,sig)
#plt.ylim(-2,2)
#plt.show()



start_index = 0
'''
slicing the data to current and future times

'''

y = sig[start_index+lags:start_index+lags+n_sample]
X = sig[start_index:start_index+n_sample]
print y.shape
print X.shape

X_container = np.zeros((lags,n_sample))
for i in range(n_sample):
    X_lags = (X[start_index+i:start_index+i+lags]).flatten()
    X_container[:,i] = X_lags


n_iter = 100
eta = 0.01
inodes = lags
onodes = 1
hnodes = 2
learning_curve = True

rnn = RNN(n_iter=n_iter,inodes=inodes,hnodes=hnodes,onodes=onodes,eta=eta,learning_curve=True,minibatches=50,lamda2=0.0,lamda1=0.0,check_gradient=True)
y = y.reshape(1,len(y))
rnn.fit(X_container,y)
#rnn.error_graph()
#rnn.draw_learning_curve()


relative_error = (abs(rnn.predict(X_container)-y).mean())/(abs(y).mean())
print "relative error",str(relative_error)


plt.plot(range(n_sample),y.flatten(),label='real data')

plt.plot(range(n_sample),rnn.predict(X_container).flatten(),label='predicted data')
plt.legend()
plt.show()



