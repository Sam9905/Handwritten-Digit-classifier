
import numpy as np
import random
try:
   import cPickle as cPickle
except:
   import pickle as cPickle

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    def printValues(self):
        print(self.weights)
        print(self.biases)
    
    def feedforward(self, a):
        #Return the output of the network if "a" is input.
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        #Train the neural network using mini-batch stochastic gradient descent.  The "training_data" is a list of tuples
        #"(x, y)" representing the (training inputs, desired outputs).  The other non-optional parameters are self-explanatory.
        if test_data: 
            #Get the size of the test data
            test_data_size = len(test_data)
        #Get size of the training data
        training_data_size = len(training_data)
        #Iterate training over all epochs
        for j in range(epochs):
            #Shuffle the Training Data
            random.shuffle(training_data)
            #Partition it into mini-batches of the appropriate size
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, training_data_size, mini_batch_size)]
            #For each mini_batch apply a single step of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            #If "test_data" is provided then the network will be evaluated against the test data after each epoch
            if test_data:
                #Print the partial progress which is useful for tracking progress, but slows things down substantially
                accuracy = self.evaluate(test_data)/test_data_size
                print ("Epoch {0} Accuracy: {1}".format(j, accuracy))
            else:
                print ("Epoch {0} Complete".format(j))
                
    def update_mini_batch(self, mini_batch, learning_rate):
        #Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #The "mini_batch" is a list of tuples "(x, y). Compute gradients for every training sample in the mini_batch
        for x, y in mini_batch:
            x= np.reshape(x, (784, 1))
            #Do the back propagation
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #Update the self.weights and self.biases appropriately.
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        #Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x.  ``nabla_b`` and
        #``nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` and ``self.weights``.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        # Return the number of test inputs for which the neural network outputs the correct result. Note that the neural
        # network's output is assumed to be the index of whichever neuron in the final layer has the highest activation.
        test_results = [(np.argmax(self.feedforward(x.reshape((784, 1)))), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    #### Miscellaneous functions
    def sigmoid(self,z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self,z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    if __name__ == '__main__':
        dlProg = DSProgress()
        dlProg.setup_data_files()
        
        # Retrieve the training, validation and test data
        training_data, validation_data, test_data = unpickle("mnist.pkl")
        display_stats(training_data, 300)
        
        net = Network([784, 100, 10])
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        training_input = training_data
        test_input = test_data
        net.SGD(training_input, 10, 20, 3, test_data=test_input)
        
        # save the model to disk
        filename = 'models/digit_rec_model.sav'
        cPickle.dump(net, open(filename, 'wb'))
        
        # load the model from disk
        loaded_model = cPickle.load(open(filename, 'rb'))
        result = loaded_model.evaluate(validation_data)
        print(result/len(validation_data))
