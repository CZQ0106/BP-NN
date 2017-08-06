import numpy as np

def tanh(x):  
    return np.tanh(x)

def tanh_deriv(x):  
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):  
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):  
    return logistic(x)*(1-logistic(x))



class NeuralNetwork:   
    def __init__(self, layers, activation='tanh'):  
        """  
        :param layers: A list containing the number of units in each layer.
        Should be at least two values  
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"  
        """  
        if activation == 'logistic':  
            self.activation = logistic  
            self.activation_deriv = logistic_derivative  
        elif activation == 'tanh':  
            self.activation = tanh  
            self.activation_deriv = tanh_deriv
    
        self.weights = []  
        for i in range(1, len(layers) - 1):  
            self.weights.append((2*np.random.random([layers[i - 1] + 1, layers[i] + 1])-1)*0.25)  
            self.weights.append((2*np.random.random([layers[i] + 1, layers[i + 1]])-1)*0.25)
        #print(self.weights)
            
    def fit(self, X, y, learning_rate=0.2, epochs=10000):         
        X = np.atleast_2d(X)         
        temp = np.ones([X.shape[0], X.shape[1]+1])         
        temp[:, 0:-1] = X  # adding the bias unit to the input layer         
        X = temp
        y = np.array(y)
        
        for k in range(epochs):  
            i = np.random.randint(X.shape[0])  
            out = [X[i]]
            
            for l in range(len(self.weights)):  #going forward network, for each layer
                out.append(self.activation(np.dot(out[l], self.weights[l])))  #Computer the node value for each layer (O_i) using activation function
            #print(out)
            error = y[i] - out[-1]  #Computer the error at the top layer
            deltas = [error * self.activation_deriv(out[-1])] #For output layer, Err calculation (delta is updated error)
            
            #Staring backprobagation
            for l in range(len(out) - 2, 0, -1): # we need to begin at the second to last layer 
                #Compute the updated error (i,e, deltas) for each node going from top layer to input layer 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(out[l]))  
            deltas.reverse()  
            for i in range(len(self.weights)):  
                layer = np.atleast_2d(out[i])  
                delta = np.atleast_2d(deltas[i])  
                self.weights[i] += learning_rate * layer.T.dot(delta)
                
                
    def predict(self, x):         
        x = np.array(x)         
        temp = np.ones(x.shape[0]+1)         
        temp[0:-1] = x         
        out = temp         
        for l in range(0, len(self.weights)):             
            out= self.activation(np.dot(out, self.weights[l]))         
        return out
nn=NeuralNetwork([2,2,1])
X=[[0,1],[1,0],[0,0],[1,1]]
y=[0,0,1,1]
nn.fit(X,y)
print(nn.predict([1,1]))

