import numpy as np

class SCR():
    def __init__(self, shape, vals, col_idxs, row_ptrs):
        self.shape = shape
        self.vals = vals
        self.col_idxs = col_idxs
        self.row_ptrs = row_ptrs

    @classmethod
    def from_dense(me, other):
        rs = len(other)
        cs = len(other[0])
        vals = []
        col_idxs = []
        row_ptrs = []
        for ri in range(rs):
            row_ptrs.append(len(vals))
            for ci in range(cs):
                if other[ri][ci] != 0:
                    col_idxs.append(ci)
                    vals.append(other[ri][ci])
        row_ptrs.append(rs+1)
        return me((rs, cs), vals, col_idxs, row_ptrs)

    def to_dense(self):
        rs = self.shape[0]
        cs = self.shape[1]
        out = [[0 for _ in range(cs)] for _ in range(rs)]
        vi = 0
        for ri in range(rs):
            # [start, end)
            start = self.row_ptrs[ri]
            end = self.row_ptrs[ri+1]
            for i in range(start, end):
                out[ri][self.col_idxs[vi]] = self.vals[vi]
                vi += 1
        return out

    def dot(self, rhs):
        pass

    def __repr__(self):
        return f"vals = {self.vals}, col_idxs = {self.col_idxs} row_ptrs = {self.row_ptrs}"
                
# creates a sparse array
def mk_arr(num_rows: int, num_cols: int):
    return [[r*num_cols+c if (r*num_cols+c) % (num_cols-1) == 0 else 0 for c in range(num_cols)] for r in range(num_rows)]


arr = mk_arr(5, 4)
print(arr)
scr = SCR.from_dense(arr)
print(scr)

np.testing.assert_allclose(arr, scr.to_dense())
