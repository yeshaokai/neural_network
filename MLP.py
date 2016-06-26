import numpy as np
import matplotlib.pyplot as plt
import copy
class MLP:
    '''
    one input one hidden one output

    '''
    def __init__(self,n_iter=100,eta=0.1,inodes=1,hnodes=1,onodes=1,check_gradient=True):
        '''
        self._w1  array, shape = [1,2],weight for hidden
        self._w2  array, shape = [1,2],weight for output
        '''
        self.inodes=inodes
        self.hnodes=hnodes
        self.onodes=onodes

        self.n_iter = n_iter
        self.eta = eta

        self.check_gradient = check_gradient

    def initialize_weights(self):
        '''
        initialize weights .
        Note that the extra one is for bias unit
        '''

        self._w1 = np.random.rand(self.hnodes,self.inodes)
        self._w2 = np.random.rand(self.onodes,self.hnodes)
        
        
    def _add_bias_unit(self,X,dir='row'):
        '''
        make the input that corresponds to bias weights 1

        '''
        if dir =='row':
            

            return X

    def net_input(self,X,W):
        '''
        return the net input of this neuron

        '''
        return np.dot(X,W)
    def _sigmoid(self,x):
        '''
        return a sigmoid function

        '''
        return 1.0/(1.0+np.exp(-x))
    def _sigmoid_gradient(self,z):
        '''
        gradient of sigmoid function

        '''
 #       sg = self._sigmoid(z)
        return z*(1-z)
    
    def feed_forward(self,x,w1,w2):
        '''

        u is the net input for hidden neuron
        
        y is the activation output from hidden neuron


        '''




        
        u = w1.dot(x)

        y = self._sigmoid(u)
        



        v = np.dot(w2,y)

        z = self._sigmoid(v)

        return x,u,y,v,z
    def get_cost(self,x,t,w1,w2):
        x,u,v,y,z=self.feed_forward(x,w1,w2)
#        print (0.5*(z-t)**2)

        return (0.5*(z-t)**2).sum()
    def numerical_gradient(self,x,t):
        '''
        numerical gradient = (J(w+episilon)-J(w-episilon))/2*episilon
        
        deep copy the weights so that the numerical evaulation does not change the value of the weights

        '''
        episilon = 1e-7
        w1=copy.deepcopy(self._w1)
        w2=copy.deepcopy(self._w2)



        gradient1 = np.zeros(w1.shape)
        for i in range(w1.shape[0]):
            for k in range(w1.shape[1]):
                w1[i][k]+=episilon
                left=self.get_cost(x,t,w1,w2)                
                w1[i][k]-=2*episilon
                right=self.get_cost(x,t,w1,w2)

                nu_gradient = (left-right)/(2*episilon)
                gradient1[i][k]=nu_gradient


        w1=copy.deepcopy(self._w1)
        w2=copy.deepcopy(self._w2)

        gradient2=np.zeros(w2.shape)
        for j in range(w2.shape[0]):            
            for i in range(w2.shape[1]):
                w2[j][i]+=episilon
                left = self.get_cost(x,t,w1,w2)
                w2[j][i]-=2*episilon
                right = self.get_cost(x,t,w1,w2)                
                nu_gradient = (left-right)/(2*episilon)
                gradient2[j][i]=nu_gradient



        return np.hstack((gradient1.flatten(),gradient2.flatten()))

    def relative_error(self,a,b):
        return np.linalg.norm(a-b)/(np.linalg.norm(a)+np.linalg.norm(b))
    def _update_weights(self,t,x,u,y,v,z):
        '''
        t [array] ->target vector
        x [array] ->input vector
        u [array] -> net input for hidden layer
        y [array] -> output vector for hidden layer
        v [array] -> net input vector for output layer
        z [array] -> output vector for output layer

        '''
        


        '''
        update rules

        self._w1 -= grad1*self.eta
        self._w2 -= grad2*self.eta

        '''
        
        output_error = z-t

        sig_grad2 = self._sigmoid_gradient(z)

        grad2 = np.dot((output_error*sig_grad2),y.T)




        step1 = self._w2.T.dot((output_error)*self._sigmoid_gradient(z))
        step2 = step1*self._sigmoid_gradient(y)
        grad1 = step2.dot(x.T)
        grad1 = grad1.reshape(2,1)






        if self.check_gradient:
            nu = self.numerical_gradient(x,t)

            ana = np.hstack((grad1.flatten(),grad2.flatten()))

#            print 'nu',str(nu)
#            print 'ana',str(ana)

            print "relative error %s" % self.relative_error(nu,ana)


        self._w2-=grad2*np.ones(grad2.shape)*self.eta
        self._w1-=grad1*np.ones(grad1.shape)*self.eta


    def fit(self,x,t):
        '''
        online learning
        

        '''
        self.error=[]
        for i in range(self.n_iter):
            error=[]
           
            w1,w2 = self._w1,self._w2
            
            x,u,y,v,z = self.feed_forward(x,w1,w2)
    
            e = self.get_cost(x,t,w1,w2)
                
            self._update_weights(t,x,u,y,v,z)
    


            self.error.append(e)


    def predict(self,x):


        x,u,y,v,z = self.feed_forward(x,self._w1,self._w2)


        return z
            
def error_graph(n_iter):
    plt.xlabel('epoch')
    plt.ylabel('sse')
    plt.scatter(np.linspace(1,n_iter,n_iter),neural.error)
#    print neural.error
def comparison_graph(X,Y):
    plt.scatter(X,Y,marker='o')
    plt.scatter(X,neural.predict(X),marker='x')


if __name__ == '__main__':
    n_iter = 400
    inodes = 1
    hnodes = 2
    onodes = 1

    neural = MLP(n_iter=n_iter,inodes=inodes,hnodes=hnodes,onodes=onodes,eta=0.1)
    n_samples = 100
    neural.initialize_weights()

    X_train = np.linspace(1,44,n_samples)*np.pi/180.
    X_train=X_train.reshape(inodes,n_samples)
    Y_train = np.sin(X_train)    
    Y_train=Y_train.reshape(onodes,n_samples)
    X_test =  np.linspace(46,89,n_samples)*np.pi/180
    X_test=X_test.reshape(inodes,n_samples)
    Y_test = np.sin(X_test)
    Y_test=Y_test.reshape(onodes,n_samples)
    neural.fit(X_train,Y_train)
    
 #   error_graph(n_iter)
#    comparison_graph(X_train,Y_train)
    comparison_graph(X_test,Y_test)
    plt.show()
    
    
