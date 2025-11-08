import functools
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

def gemm_tiled(lhs, rhs, tiling):
    assert len(lhs[0]) == len(rhs) 
    m, k, n = len(lhs), len(lhs[0]), len(rhs[0])
    mt, kt, nt = tiling
    assert m % mt == 0
    assert k % kt == 0
    assert n % nt == 0
    out = [[0 for _ in range(n)] for _ in range(m)]
    # iterate over the tiles
    for mti in range(0, m, mt):
        for nti in range(0, n, nt):
            for kti in range(0, k, kt):
                # iterate over the elements in the tile
                for mi in range(mti, mti+mt):
                    # row of output
                    out_mi = out[mi]
                    for ni in range(nti, nti+nt):
                        acc = 0
                        for ki in range(kti, kti+kt):
                            acc += lhs[mi][ki] * rhs[ki][ni]
                        out_mi[ni] += acc
    return out

gemm_tiled_8_16_32 = functools.partial(gemm_tiled, tiling=(8, 16, 32))
gemm_tiled_8_16_32.__name__ = "gemm_tiled_8_16_32"
print(time_fn(gemm_tiled_8_16_32, x, y))

gemm_tiled_32_32_32 = functools.partial(gemm_tiled, tiling=(32, 32, 32))
gemm_tiled_32_32_32.__name__ = "gemm_tiled_32_32_32"
print(time_fn(gemm_tiled_32_32_32, x, y))
