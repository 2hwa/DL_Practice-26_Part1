import numpy as np
import functions as f
import Layers as l
import optimizers as o
from collections import OrderedDict
#affine1 - batchnorm - relu1 - affine2 -batchnorm - Relu2 - .. - affine 5 - swl

class MulLayerNet:
    def __init__(self):
        self.optimizer = o.Adam()
        self.params = {}
        all_size_list = [self.input_size] + self.hidden_list + [self.output_size]
        
        for idx in range(1, len(all_size_list)):
        # He 초깃값 적용
            scale = np.sqrt(2.0 / all_size_list[idx-1]) 
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

        self.grads = {}

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        self.layers = OrderedDict()
        self.layers['Affine1'] = l.Affine(W1, b1)
        self.layers['Relu1'] = l.Relu()
        self.layers['Affine2'] = l.Affine(W2, b2)
        self.lastLayer = l.softmaxwithloss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    
    def run(self, x, t, input_size, hidden_list, output_size):
        # Update the network architecture with the provided dimensions
        self.input_size = input_size
        self.hidden_list = hidden_list
        self.output_size = output_size

        # Reinitialize the network with the new dimensions
        self.__init__()

        loss_val = self.loss(x,t)
        dout = 1
        dout = self.lastLayer.backward(dout)

        for i in reversed(list(self.layers.values())):
            dout = i.backward(dout)

        self.grads['W1'] = self.layers['Affine1'].dW
        self.grads['b1'] = self.layers['Affine1'].db
        self.grads['W2'] = self.layers['Affine2'].dW
        self.grads['b2'] = self.layers['Affine2'].db

        self.optimizer.update(self.params, self.grads)

        return loss_val

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
