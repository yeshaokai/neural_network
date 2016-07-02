import pandas as pd
from MLP import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
import numpy as np

import sys

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)


dum = df.loc[:,:].values
le = LabelEncoder()
enc = OneHotEncoder(categorical_features=[4])
dum[:,4]=le.fit_transform(dum[:,4])

dum= enc.fit_transform(dum).toarray()
X=  dum[:,3:]
y = dum[:,0:3]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

kfold = StratifiedKFold(y = df.loc[:,4],n_folds = 10,random_state=1)

scores = []
pca = PCA(n_components = 3)
for k,(train,test) in enumerate(kfold):    
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]

    stdsc = StandardScaler()
    
    X_train_std = stdsc.fit_transform(X_train)    
    X_test_std = stdsc.fit_transform(X_test)        

    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    n_iter = 1000
    inodes = 3
    hnodes = 3
    onodes = 3    

    mlp =MLP(n_iter=n_iter,inodes=inodes,hnodes=hnodes,onodes=onodes,eta=0.2,learning_curve=True)
    mlp.fit(X_train_pca.T,y_train.T)        
    y_predict = mlp.predict(X_test_pca.T)    
    y_predict = np.where(y_predict>=0.5,1,0)    
    count = 0
    miscount = 0
    for i in range(y_predict.shape[1]):
        if np.array_equal(y_predict[:,i],y_test.T[:,i]):
            count +=1
        else:
            miscount +=1
    scores.append(count*100.0/y_predict.shape[1])

    print 'Fold: %s, Score %s' % (k,scores[-1]) 
    print 'Fold: %s, misclassified %s ' %(k,miscount)

print 'average is %s '%(np.mean(scores))
mlp.draw_learning_curve()
