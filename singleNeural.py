import numpy as np
import matplotlib.pyplot as plt
import copy
class SingleNeural:
    '''
    one input one hidden one output

    '''
    def __init__(self,n_iter=100,eta=0.1,check_gradient=False):
        '''
        self._w1  array, shape = [1,2],weight for hidden
        self._w2  array, shape = [1,2],weight for output
        '''
        self.n_iter = n_iter
        self.eta = eta
        self._w1 = np.random.rand(2)
        self._w2 = np.random.rand(2)
        self.check_gradient = check_gradient


    def _add_bias_unit(self,X):
        '''
        make the input that corresponds to bias weights 1

        '''
        l = X.shape[0]
        new_X = np.ones((1,l+1))
        new_X[1:]=X
        return new_X

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
    def sigmoid_gradient(self,z):
        '''
        gradient of sigmoid function

        '''
        sg = self._sigmoid(z)
        return sg*(1-sg)
    
    def feed_forward(self,x,w1,w2):
        '''

        u is the net input for hidden neuron
        
        y is the activation output from hidden neuron


        '''

        x_bias = 1
        u = np.dot(w1,np.array([x_bias,x]))
        y = self._sigmoid(u)
        y_bias = 1
        v = np.dot(w2,np.array([y_bias,y]))
        z = self._sigmoid(v)
        return x,u,y,v,z
    def nu_gradient(self,x,t,w_str):
        '''
        numerical gradient = (J(w+episilon)-J(w-episilon))/2*episilon
        
        deep copy the weights so that the numerical evaulation does not change the value of the weights

        '''
        episilon = 1e-7
        w1=copy.deepcopy(self._w1.reshape(2))
        w2=copy.deepcopy(self._w2.reshape(2))
        
        if w_str == 'a0':

            w1[0]+=episilon
            x,u,y,v,z0 = self.feed_forward(x,w1,w2)
            w1[0]-=2*episilon
            x,u,y,v,z1 = self.feed_forward(x,w1,w2)
            left = 0.5*(z0-t)**2
            right = 0.5*(z1-t)**2
            return (left-right)/(2*episilon)
           
        if w_str == 'a1':
            w1[1]+=episilon
            x,u,y,v,z0 = self.feed_forward(x,w1,w2)
            w1[1]-=2*episilon
            x,u,y,v,z1 = self.feed_forward(x,w1,w2)
            left = 0.5*(z0-t)**2
            right = 0.5*(z1-t)**2
            return (left-right)/(2*episilon)

        if w_str == 'b0':

            w2[0]+=episilon
            x,u,y,v,z0 = self.feed_forward(x,w1,w2)
            w2[0]-=2*episilon
            x,u,y,v,z1 = self.feed_forward(x,w1,w2)
            left = 0.5*(z0-t)**2
            right = 0.5*(z1-t)**2
            return (left-right)/(2*episilon)


        if w_str == 'b1':
            w2[1]+=episilon
            x,u,y,v,z0 = self.feed_forward(x,w1,w2)
            w2[1]-=2*episilon
            x,u,y,v,z1 = self.feed_forward(x,w1,w2)
            left = 0.5*(z0-t)**2
            right = 0.5*(z1-t)**2
            return (left-right)/(2*episilon)


    def relative_error(self,a,b):
        return np.linalg.norm(a-b)/(np.linalg.norm(a)+np.linalg.norm(b))
    def _update_weights(self,t,x,u,y,v,z):
        '''
        update weights based on error
        
        '''
        #error = 0.5*((z-y)**2)
        p = (z-t)*z*(1-z)
        '''
        update rules

        self._w1 -= grad1*self.eta
        self._w2 -= grad2*self.eta

        '''
        
        
        grad2 = np.array([p,p*y])

        grad1 = np.array([p*self._w2[1]*y*(1-y),p*self._w2[1]*y*(1-y)*x])


        if self.check_gradient:
            nu = np.array([self.nu_gradient(x,t,'a0'),self.nu_gradient(x,t,'a1'),self.nu_gradient(x,t,'b0'),self.nu_gradient(x,t,'b1')]).reshape(4)
            ana = np.array([grad1[0],grad1[1],grad2[0],grad2[1]]).reshape(4)
            print "relative error %s" % self.relative_error(nu,ana)

        self._w2-=grad2*self.eta
        self._w1-=grad1*self.eta


    def fit(self,X,T):
        '''
        online learning
        

        '''
        self.error=[]
        for i in range(self.n_iter):
            error=[]
            for x,t in zip(X,T):
                w1,w2 = self._w1,self._w2
                x,u,y,v,z = self.feed_forward(x,w1,w2)
                e = 0.5*((z-t)**2)
                self._update_weights(t,x,u,y,v,z)
                error.append(e)
            self.error.append(np.mean(error))


    def predict(self,X):
        array = []
        for x in X:
            x,u,y,v,z = self.feed_forward(x,self._w1,self._w2)
            array.append(z)

        return array
            
def error_graph(n_iter):
    plt.xlabel('epoch')
    plt.ylabel('sse')
    plt.scatter(np.linspace(1,n_iter,n_iter),neural.error)

def comparison_graph(X,Y):
    plt.scatter(X,Y,marker='o')
    plt.scatter(X,neural.predict(X),marker='x')

if __name__ == '__main__':
    n_iter = 400
    neural = SingleNeural(n_iter=n_iter,eta=0.6)
    X_train = np.linspace(1,44,100)*np.pi/180.
    Y_train = np.sin(X_train)    
    X_test =  np.linspace(46,89,100)*np.pi/180
    Y_test = np.sin(X_test)
    neural.fit(X_train,Y_train)
    
#    error_graph(n_iter)
    comparison_graph(X_train,Y_train)
    plt.show()
    
    
