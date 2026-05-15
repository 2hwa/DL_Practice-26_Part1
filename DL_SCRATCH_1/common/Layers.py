import numpy as np
import functions as f

class Relu:
    def __init__(self):
        self.mask = None #실수로 backward 먼저수행시 error.. 방지용 + 변수관리

    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0 #out[] << 인덱싱임!

        return out
    
    def backward(self,dout):
        dout[self.mask] = 0 #forward를 무조건 작동시킨 상태에서 backpropagation하기 때문에 self.mask에 적용되어잇음.
        dx = dout
        return dx




class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        #역전파용
        self.originalx = None
        self.dW = None
        self.db = None

    def forward(self, x):
        #역전파용
        self.originalx = x.shape

        #Affine구현
        x = x.reshape(x.shape[0],-1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b #중 ! 요

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.originalx) 
        #역전파용
        #원래의 x의 shape로 되돌려주는 코드. *는 언팩연산자, 
        # #self.originalx는 튜플이니까 언팩연산자로 풀어서 reshape에 넣어주는 것.

        return dx

class softmaxwithloss:
      def __init__(self):
            self.y = None
            self.t = None

      def forward(self,a,t):
            self.t = t
            self.y = f.softmax(a)
            self.loss = f.cee(self.y, self.t)

            return self.loss

      def backward(self, dout=1):
           batch_size = self.t.shape[0]
           dx = (self.y - self.t)/batch_size

           return dx

class batchnorm:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None 

        # 시험 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # 역전파 시 필요한 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        if self.running_mean is None:
            D = x.shape[1]
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        # 역전파 구현 (조금 복잡하지만 공식대로 구현)
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std**2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask



im2col = f.im2col
col2im = f.col2im

class Conv:
    def __init__ (self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 역전파 시 필요한 중간 데이터
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
        
        


    




#test code
#swl = softmaxwithloss()
#a = np.array([[0.3, 2.9, 4.0], [0.1, 0.2, 0.3]])
#t = np.array([[0, 0, 1], [0, 1, 0]])
#print(swl.forward(a,t))

