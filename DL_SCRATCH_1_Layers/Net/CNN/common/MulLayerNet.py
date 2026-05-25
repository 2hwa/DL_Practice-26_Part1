import numpy as np
import functions as f
import Layers as l
import optimizers as o
from collections import OrderedDict
#affine1 - batchnorm - relu1 - affine2 -batchnorm - Relu2 - .. - affine 5 - swl



class MulLayerNet:
    def __init__(self, input_size=784, hidden_list=[500, 200, 90, 35], output_size=10):
        self.optimizer = o.Adam()
        self.params = {}
        self.grads = {}
        self.layers = OrderedDict()
        self.input_size = input_size
        self.hidden_list = hidden_list
        self.output_size = output_size

        all_size_list = [self.input_size] + self.hidden_list + [self.output_size]

        Affine, Relu, SoftmaxWithLoss = l.Affine, l.Relu, l.softmaxwithloss
        batchnorm = l.batchnorm
        Dropout = l.Dropout

        for idx in range(1, len(all_size_list)):
        # He 초깃값 적용
            scale = np.sqrt(2.0 / all_size_list[idx-1]) 
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

        for idx in range(1, len(all_size_list) - 1):
             self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
             self.layers['BatchNorm' + str(idx)] = batchnorm(gamma=1.0, beta=0.0)
             self.layers['ReLU' + str(idx)] = Relu()
             self.layers['Dropout' + str(idx)] = Dropout(dropout_ratio=0.15)


        last_idx = len(all_size_list) - 1
        self.layers['Affine' + str(last_idx)] = Affine(self.params['W' + str(last_idx)], self.params['b' + str(last_idx)])

        self.lastLayer = SoftmaxWithLoss()




    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    
    def run(self, x, t):

        loss_val = self.loss(x,t)
        dout = 1
        dout = self.lastLayer.backward(dout)

        for i in reversed(list(self.layers.values())):
            dout = i.backward(dout)

        for j in range(1, len(self.hidden_list) + 2):
             self.grads['W' + str(j)] = self.layers['Affine' + str(j)].dW
             self.grads['b' + str(j)] = self.layers['Affine' + str(j)].db

        self.optimizer.update(self.params, self.grads)

        return loss_val

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
