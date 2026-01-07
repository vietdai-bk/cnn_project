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

x = np.random.rand(10, 10)
k = np.random.rand(3, 3)
y = conv2d(x, k, stride=2, padding=1)
print(y.shape) 
# output = (5, 5)
# H = (10 + 2 * 1 - 3)//2 + 1 = 9//2 + 1 = 5
# W = H