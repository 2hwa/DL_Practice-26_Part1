import numpy as np
from common import functions as f
from common import Layers as l
from common import optimizers as o
from collections import OrderedDict

Affine1 = l.Affine()
Affine2 = l.Affine()
Relu1 = l.Relu()
swl = l.softmaxwithloss()
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

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        self.layers = OrderedDict()
        self.layers['Affine1'] = l.Affine(W1, b1)
        self.layers['Relu1'] = l.Relu()
        self.layers['Affine2'] = l.Affine(W2, b2)
        self.lastLayer = l.softmaxwithloss() # 이 코드에서 swl가 이미 잇는데 굳이 필요한가..?


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return swl.forward(y,t)
    
    def modify(self, x, t):
        self.loss(x,t)

        dout = 1                             #for문으로 수정필요
        dout = swl.backward(dout)
        dout = Affine2.backward(dout)
        dout = Relu1.backward(dout)
        dout = Affine1.backward(dout)
        SGD.update(self.params, Affine1.dW)
        SGD.update(self.params, Affine1.db)
        SGD.update(self.params, Affine2.dW)
        SGD.update(self.params, Affine2.db)

        

    

    
