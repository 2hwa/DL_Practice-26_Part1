import numpy as np

def softmax(self, a):
        a_max = np.max(a, axis = 1, keepdims=True) #overflow 방지대책
        exp_a = np.exp(a-a_max)
        sum_exp_a = np.sum(exp_a, axis = 1, keepdims=True) # axis = 1, keepdims 잘 생각해보기 / axis = 1 은 행 기준, axis =0 는 열 기준
        
        y = exp_a / sum_exp_a


        return y
    
def cee(self, y, t): #cross entropy error (CEE)
    batch_size = self.y.shape[0] # col 기준 = 행의 갯수
    loss = -np.sum(self.t*np.log(self.y + 1e-7))/batch_size

    return loss



            
