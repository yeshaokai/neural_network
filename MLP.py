import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
class MLP:
    '''
    one input one hidden one output

    '''
    def __init__(self,n_iter=100,lamda1=0.0000,lamda2=0,eta=0.1,inodes=1,hnodes=1,onodes=1,check_gradient=True):
        '''
        self._w1  array, shape = [1,2],weight for hidden
        self._w2  array, shape = [1,2],weight for output
        '''
        self.n_sample=0
        self.inodes=inodes
        self.hnodes=hnodes
        self.onodes=onodes
        self.n_iter = n_iter
        self.eta = eta
        self.lamda2 = lamda2
        self.lamda1 = lamda1
        self.check_gradient = check_gradient

        self._initialize_weights()
    def _initialize_weights(self):
        '''
        initialize weights .
        Note that the extra one is for bias unit
        '''

        self._w1 = np.random.uniform(-1,1,(self.hnodes,self.inodes))
        self._w2 = np.random.uniform(-1,1,(self.onodes,self.hnodes))
        self._bias1 = np.random.uniform(-1,1,(self.hnodes,1))
        self._bias2 = np.random.uniform(-1,1,(self.onodes,1))
        


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
    
    def feed_forward(self,x,w1,w2,bias1,bias2):
        '''

        u is the net input for hidden neuron
        
        y is the activation output from hidden neuron


        '''


        u = w1.dot(x)
 #       u += bias1.dot(np.ones(u.shape))

        u+= bias1*np.ones(u.shape)
       





        y = self._sigmoid(u)

        v = np.dot(w2,y)

        v+= bias2*np.ones(v.shape)



        z = self._sigmoid(v)

        return x,u,y,v,z
    def L2_term(self,w1,w2):
        
        return 0.5*self.lamda2*((w1.flatten()**2).sum()+(w2.flatten()**2).sum())
    def L1_term(self,w1,w2):
        return 0.5*self.lamda1*(np.abs(w1.flatten()).sum()+np.abs(w2.flatten().sum()))

    def get_cost(self,x,t,w1,w2,bias1,bias2):
        x,u,v,y,z=self.feed_forward(x,w1,w2,bias1,bias2)

        


        return (0.5*(z-t)**2+self.L2_term(w1,w2)+self.L1_term(w1,w2)).sum()

    def numerical_gradient(self,x,t):
        '''
        numerical gradient = (J(w+episilon)-J(w-episilon))/2*episilon
        
        deep copy the weights so that the numerical evaulation does not change the value of the weights

        '''
        episilon = 1e-6
        w1=copy.deepcopy(self._w1)
        w2=copy.deepcopy(self._w2)
        


        gradient1 = np.zeros(w1.shape)
        for i in range(w1.shape[0]):
            for k in range(w1.shape[1]):
                w1[i][k]+=episilon
                left=self.get_cost(x,t,w1,w2,self._bias1,self._bias2)                
                w1[i][k]-=2*episilon
                right=self.get_cost(x,t,w1,w2,self._bias1,self._bias2)

                nu_gradient = (left-right)/(2*episilon)
                gradient1[i][k]=nu_gradient


        w1=copy.deepcopy(self._w1)
        w2=copy.deepcopy(self._w2)

        gradient2=np.zeros(w2.shape)
        for j in range(w2.shape[0]):            
            for i in range(w2.shape[1]):
                w2[j][i]+=episilon
                left = self.get_cost(x,t,w1,w2,self._bias1,self._bias2)
                w2[j][i]-=2*episilon
                right = self.get_cost(x,t,w1,w2,self._bias1,self._bias2)                
                nu_gradient = (left-right)/(2*episilon)
                gradient2[j][i]=nu_gradient

        bias1 = copy.deepcopy(self._bias1)
        bias2 = copy.deepcopy(self._bias2)

        bias1_gradient = np.zeros(self.hnodes)
        for i in range(bias1_gradient.shape[0]):
            bias1[i] +=episilon
            left = self.get_cost(x,t,self._w1,self._w2,bias1,bias2)
            bias1[i] -=2*episilon
            right = self.get_cost(x,t,self._w1,self._w2,bias1,bias2)
            bias1_gradient[i] = (left-right)/(2*episilon)
        bias1_gradient = bias1_gradient.reshape(self.hnodes,1)

        bias1 = copy.deepcopy(self._bias1)
        bias2 = copy.deepcopy(self._bias2)
        
        bias2_gradient = np.zeros(self.onodes)
        for i in range(bias2_gradient.shape[0]):

            bias2[i] +=episilon
            left = self.get_cost(x,t,self._w1,self._w2,bias1,bias2)
            bias2[i] -=2*episilon
            right = self.get_cost(x,t,self._w1,self._w2,bias1,bias2)
            bias2_gradient[i] = (left-right)/(2*episilon)
        bias2_gradient.reshape((self.onodes,1))

        num_weight_gradient = self._w1.shape[0]*self._w1.shape[1]+self._w2.shape[0]*self._w2.shape[1]

        num_bias_gradient = self._bias1.shape[0]+self._bias2.shape[0]

        gradient_array = np.zeros((1,num_weight_gradient+num_bias_gradient))

        weight_gradient = np.hstack((gradient1.flatten(),gradient2.flatten()))

        bias_gradient = np.hstack((bias1_gradient.flatten(),bias2_gradient.flatten()))

        bias_gradient = bias_gradient.reshape(1,num_bias_gradient)




        gradient_array[:,0:num_weight_gradient]+=weight_gradient

        gradient_array[:,num_weight_gradient:]+=bias_gradient
        return gradient_array
    
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
 
        grad2 +=self.lamda2*(self._w2)+self.lamda1*1


        step1 = self._w2.T.dot((output_error)*self._sigmoid_gradient(z))
        step2 = step1*self._sigmoid_gradient(y)
        grad1 = step2.dot(x.T)        

        grad1+=self.lamda2*(self._w1)+self.lamda1*1

        grad1 = grad1.reshape(self._w1.shape)



        bias2_gradient = (output_error*sig_grad2).dot(np.ones((self.n_sample,1)))
        bias1_gradient = (step1*self._sigmoid_gradient(y)).dot(np.ones((self.n_sample,1)))
        if self.check_gradient:
            nu = self.numerical_gradient(x,t)
            ana = np.hstack((grad1.flatten(),grad2.flatten()))
            ana = np.hstack((ana,bias1_gradient.flatten()))
            ana = np.hstack((ana,bias2_gradient.flatten()))




#            print "relative error %s" % self.relative_error(nu,ana)

        self._bias2 -= (output_error*sig_grad2).dot(np.ones((self.n_sample,1)))*self.eta        
        self._bias1 -= (step1*self._sigmoid_gradient(y)).dot(np.ones((self.n_sample,1)))*self.eta

        self._w2-=grad2*np.ones(grad2.shape)*self.eta
        self._w1-=grad1*np.ones(grad1.shape)*self.eta

    def fit(self,x,t):
        '''
        online learning
        
        
        '''
        self.n_sample = x.shape[1]
        self.error=[]
        for i in range(self.n_iter):
            error=[]
           
#            w1,w2 = self._w1,self._w2

            x,u,y,v,z = self.feed_forward(x,self._w1,self._w2,self._bias1,self._bias2)
    
            e = self.get_cost(x,t,self._w1,self._w2,self._bias1,self._bias2)
                
            self._update_weights(t,x,u,y,v,z)
    


            self.error.append(e)


    def predict(self,x):


        x,u,y,v,z = self.feed_forward(x,self._w1,self._w2,self._bias1,self._bias2)


        return z
            
def error_graph(n_iter,neural):
    plt.xlabel('epoch')
    plt.ylabel('sse')
    plt.plot(np.linspace(1,n_iter,n_iter),neural.error)
    plt.show()
def comparison_graph(X,Y):
    plt.scatter(X,Y,marker='o',label='real data',color='yellow')
    plt.scatter(X,neural.predict(X),marker='x',label='network output',color='red')
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    n_iter = 400
    inodes = 1
    hnodes = 1
    onodes = 1

    neural = MLP(n_iter=n_iter,inodes=inodes,hnodes=hnodes,onodes=onodes,eta=0.2)
    n_samples = 100


    X_train = np.linspace(1,30,n_samples)*np.pi/180.
    X_train=X_train.reshape(inodes,n_samples)
    Y_train = np.sin(X_train)    
    Y_train=Y_train.reshape(onodes,n_samples)
    '''
    X_test =  np.linspace(44,89,n_samples)*np.pi/180
    X_test=X_test.reshape(inodes,n_samples)
    Y_test = np.sin(X_test)
    Y_test=Y_test.reshape(onodes,n_samples)
    '''
    neural.fit(X_train,Y_train)
    
#    error_graph(n_iter,neural)
#    comparison_graph(X_test,Y_test)
    comparison_graph(X_train,Y_train)

    
    