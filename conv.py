import numpy as np
import scipy

import gemm

def conv_2d(arr, kernel, padding = 0, stride = 1):
    arr_height = len(arr)
    arr_width = len(arr[0])
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])

    assert padding >= 0
    padded_arr = arr
    if padding > 0:
        padded_height = arr_height + 2 * padding
        padded_width = arr_width + 2 * padding
        padded_arr = [[0.0 for _ in range(padded_width)] for _ in range(padded_height)]
        for i in range(arr_height):
            for j in range(arr_width):
                padded_arr[padding+i][padding+j] = arr[i][j]

    # assuming no padding and stride of 1.
    out_height = (len(padded_arr) - kernel_height) // stride + 1
    out_width = (len(padded_arr[0]) - kernel_width) // stride + 1
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

def im2col(arr, kernel_h, kernel_w):
    arr_h = len(arr)
    arr_w = len(arr[0])
    out_h = arr_h - kernel_h + 1
    out_w = arr_w - kernel_w + 1
    out = []
    for i in range(out_h):
        for j in range(out_w):
            curr = []
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    curr.append(arr[i+ki][j+kj])
            out.append(curr)
    return out


print(im2col(x, k_height, k_width))


def conv2d_as_matmul(arr, kernel):
    arr_h = len(arr)
    arr_w = len(arr[0])
    kernel_h = len(kernel)
    kernel_w = len(kernel[0])
    out_h = arr_h - kernel_h + 1
    out_w = arr_w - kernel_w + 1
    # flatten arr to rows
    arr = im2col(arr, kernel_h, kernel_w)
    print(f"arr = {arr}")
    # flatten kernel to cols
    flattened = [x for row in kernel for x in row]
    print(f"flattened = {flattened}")
    xkernel = [[x] for x in flattened]
    print(f"xkernel = {xkernel}")
    out = gemm.gemm_ijk(arr, xkernel)
    print(f"dot = {out}")
    # reshape
    result = []
    i = 0
    for _ in range(out_h):
        row = []
        for _ in range(out_w):
            row.append(out[i][0])
            i += 1
        result.append(row)
    return result

out_as_matmul = conv2d_as_matmul(x, k)
print(out_as_matmul)
np.testing.assert_allclose(out, out_as_matmul)
