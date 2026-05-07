import numpy as np
from common import functions as f
from common import Layers as l
from common import optimizers as o
from collections import OrderedDict

Affine1 = l.Affine()
Affine2 = l.Affine()
Relu1 = l.Relu()
SGD = o.SGD()
Momentum = o.Momentum()
Adam = o.Adam()

#affine1 - relu1 - affine2 -swl

class TwoLayerNet:
    def __init__(self):
        self.params = {}
        self.params['W1'] = np.random.rand(784,100) #가중치 초기화
        self.params['b1'] = np.random.rand(100) #편향 초기화
        self.params['W2'] = np.random.rand(100,10)
        self.params['b2'] = np.random.rand(10)
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
    
    def modify(self, x, t):
        self.loss(x,t)
        dout = 1
        dout = self.lastLayer.backward(dout)

        for i in reversed(list(self.layers.values())):
            dout = i.backward(dout)
            
        self.grads['W1'] = self.layers['Affine1'].dW
        self.grads['b1'] = self.layers['Affine1'].db
        self.grads['W2'] = self.layers['Affine2'].dW
        self.grads['b2'] = self.layers['Affine2'].db

        Adam.update(self.params, self.grads)




        

    

    
