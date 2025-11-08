import time
import numpy as np

def time_fn(fn, lhs, rhs):
    start = time.perf_counter()
    out = fn(lhs, rhs)
    duration = time.perf_counter() - start
    # test correctness against numpy
    np.testing.assert_allclose(out, np.dot(np.array(lhs), np.array(rhs)))
    return duration, fn.__name__

def mk_arr(num_rows: int, num_cols: int):
    return [[r*num_cols+c for c in range(num_cols)] for r in range(num_rows)]

print(mk_arr(5, 4))


def gemm_ijk(lhs, rhs):
    assert len(lhs[0]) == len(rhs) 
    m, k, n = len(lhs), len(lhs[0]), len(rhs[0])
    out = [[0 for _ in range(n)] for _ in range(m)]
    for mi in range(m):
        for ni in range(n):
            for ki in range(k):
                out[mi][ni] += lhs[mi][ki] * rhs[ki][ni]
    return out

x = mk_arr(128, 256)
y = mk_arr(256, 512)

print(time_fn(gemm_ijk, x, y))

def gemm_ikj(lhs, rhs):
    assert len(lhs[0]) == len(rhs) 
    m, k, n = len(lhs), len(lhs[0]), len(rhs[0])
    out = [[0 for _ in range(n)] for _ in range(m)]
    for mi in range(m):
        for ki in range(k):
            lhs_mi_ki = lhs[mi][ki]
            # row of rhs
            rhs_ki = rhs[ki]
            # row of output
            out_mi = out[mi]
            for ni in range(n):
                out_mi[ni] += lhs_mi_ki * rhs_ki[ni]
    return out

print(time_fn(gemm_ikj, x, y))
