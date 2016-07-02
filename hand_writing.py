import struct 
import os 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA

from MLP import *
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte' 
                                % kind)
    images_path = os.path.join(path, 
                               '%s-images-idx3-ubyte' 
                               % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
 
    return images, labels
X_train, y_train = load_mnist('mnist', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('mnist', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
enc = OneHotEncoder(categorical_features=[0])
y_train  = y_train.reshape(len(y_train),1)
y_test   = y_test.reshape(len(y_test),1)


y_train = enc.fit_transform(y_train).toarray()
y_test = enc.fit_transform(y_test).toarray()

'''
constrain the size of the problem

'''
'''
train_range = range(10000)
test_range = range(1000)
y_train = y_train[train_range]
y_test = y_test[test_range]
X_train = X_train[train_range,:]
X_test = X_test[test_range,:]
'''

print X_train.shape
pca = PCA(n_components = 700)

X_train = pca.fit_transform(X_train)
print X_train.shape
X_test = pca.transform(X_test)
print "finish pca"
inodes = 700 #X_train.shape[1]
hnodes = 400
onodes = 10
n_iter = 1500
eta = 0.001
minibatches = 200
lamda2 = 0.1
lamda1 = 0.00

mlp = MLP(inodes = inodes, hnodes = hnodes , onodes = onodes, eta  = eta ,n_iter=n_iter,minibatches=minibatches,lamda2=lamda2,lamda1=lamda1,learning_curve=True)

print "hnodes %s n_iter %s eta %s minibatch %s lamda2 %s lamda1 %s" %(hnodes,n_iter,eta,minibatches,lamda2,lamda1)

mlp.fit(X_train.T,y_train.T)

#error_graph(n_iter,mlp)

y_predict = mlp.predict(X_test.T)
y_original = y_predict
y_predict = np.where(y_predict>=0.5,1,0)

count = 0
miscount = 0
for i in range(y_predict.shape[1]):
    if np.array_equal(y_predict[:,i],y_test.T[:,i]):
        count +=1
    else:

        miscount +=1

print "correct classified",str(count)
print "misclassfied",str(miscount)
#print y_predict
mlp.draw_learning_curve()
