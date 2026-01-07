import numpy as np

def conv2d(input, kernel, stride=1, padding=0):
    H, W = input.shape
    kH, kW = kernel.shape
    input_padded = np.pad(input, padding)
    out_h = (H + 2*padding - kH)//stride + 1
    out_w = (W + 2*padding - kW)//stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = input_padded[
                i*stride:i*stride+kH,
                j*stride:j*stride+kW
            ]
            output[i, j] = np.sum(region * kernel)
    return output

def max_pool2d(input, pool_size=2, stride=2):
    H, W = input.shape
    pH, pW = pool_size, pool_size
    out_h = (H - pH) // stride + 1
    out_w = (W - pW) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = input[
                i*stride:i*stride+pH,
                j*stride:j*stride+pW
            ]
            output[i, j] = np.max(region)
    return output

def avg_pool2d(input, pool_size=2, stride=2):
    H, W = input.shape
    pH, pW = pool_size, pool_size
    out_h = (H - pH) // stride + 1
    out_w = (W - pW) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = input[
                i*stride:i*stride+pH,
                j*stride:j*stride+pW
            ]
            output[i, j] = np.mean(region)
    return output

x = np.random.randint(0, 3, size=(10,10))
k = np.array([   ## laplacian filter
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
])
print(x)
y = conv2d(x, k, stride=1, padding=1)
print(y)
y_max = max_pool2d(y, pool_size=2)
print(y_max)
y_avg = avg_pool2d(y, pool_size=2)
print(y_avg)
print("Conv2d Output: ", y.shape)
print("MaxPooling Output: ", y_max.shape)
print("AveragePooling Output: ", y_avg.shape)