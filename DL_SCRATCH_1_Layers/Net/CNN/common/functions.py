import numpy as np

def softmax(a):
        a_max = np.max(a, axis = 1, keepdims=True) #overflow 방지대책
        exp_a = np.exp(a-a_max)
        sum_exp_a = np.sum(exp_a, axis = 1, keepdims=True) # axis = 1, keepdims 잘 생각해보기 / axis = 1 은 행 기준, axis =0 는 열 기준
        
        y = exp_a / sum_exp_a


        return y
    
def cee(y, t): #cross entropy error (CEE)
    batch_size = y.shape[0] # col 기준 = 행의 갯수
    loss = -np.sum(t*np.log(y + 1e-7))/batch_size

    return loss

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)

    img = np.zeros((N, C, H + 2*pad + stride -1, W + 2*pad + stride -1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]