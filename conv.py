import numpy as np
import scipy

def conv_2d(arr, kernel):
    arr_height = len(arr)
    arr_width = len(arr[0])
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])

    # assuming no padding and stride of 1.
    out_height = arr_height - kernel_height + 1
    out_width = arr_width - kernel_width + 1
    out = [[0.0 for _ in range(out_width)] for _ in range(out_height)]

    for i in range(out_height):
        for j in range(out_width):
            dot_prod = 0.0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    dot_prod += arr[i+ki][j+kj] * kernel[ki][kj]
            out[i][j] = dot_prod
    return out


x_height, x_width = 5, 4
x = [[h*x_width+w for w in range(x_width)] for h in range(x_height)]
print(x)

k_height, k_width = 3, 2
k = [[h*k_width+w for w in range(k_width)] for h in range(k_height)]
print(k)

out = conv_2d(x, k)
print(out)

np.testing.assert_allclose(scipy.signal.correlate2d(np.array(x), np.array(k), mode='valid'), out)
