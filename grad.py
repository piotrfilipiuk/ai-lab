import jax
import jax.numpy as jnp

class Value:
    def __init__(self, data, operands=(), name=""):
        self.data = data
        self.grad = 0
        self.operands = operands
        self.name = name
        self.bwd = lambda: None

    def __repr__(self):
        return f"data={self.data}, grad={self.grad}"

    def __add__(self, rhs):
        rhs = rhs if isinstance(rhs, Value) else Value(rhs)
        out = Value(self.data + rhs.data, (self, rhs), "+")
        def bwd():
            self.grad += out.grad
            rhs.grad += out.grad
        out.bwd = bwd
        return out

    def __mul__(self, rhs):
        rhs = rhs if isinstance(rhs, Value) else Value(rhs)
        out = Value(self.data * rhs.data, (self, rhs), "*")
        def bwd():
            self.grad += rhs.data * out.grad
            rhs.grad += self.data * out.grad
        out.bwd = bwd
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), f"expecting int or float, got {type(power)}"
        out = Value(self.data ** power, (power,), "**")
        def bwd():
            self.grad += (power * self.data ** (power-1)) * out.grad
        out.bwd = bwd
        return out

    def relu(self):
        out = Value(0 if self.data <= 0 else self.data, (self,), "ReLU")
        def bwd():
            self.grad += (0 if self.data == 0 else 1) * out.grad
        out.bwd = bwd
        return out

    def backward(self):

        top = []
        visited = set()
        def dfs(curr):
            visited.add(curr)
            print(curr)
            for operand in curr.operands:
                if operand in visited:
                    continue
                dfs(operand)
            top.append(curr)
        dfs(self)
        self.grad = 1
        print("start")
        for value in top[::-1]:
            print(value)
            value.bwd()

    def __neg__(self):
        return self * (-1)

    def __sub__(self, rhs):
        return self + (-rhs)

    def __rsub__(self, lhs):
        return lhs + (-self)

    def __radd__(self, lhs):
        return self + lhs

    def __rmul__(self, lhs):
        return self * lhs

    def __truediv__(self, rhs):
        return self * rhs**-1

    def __rtruediv__(self, lhs):
        return lhs * self**-1
        

tol = 1e-7
def assert_close(a, b):
    abs_diff = abs(a-b)
    assert abs_diff < tol, f"{a} vs {b}, abs_diff = {abs_diff}"

def fwd(x):
    x1 = x + 2
    x2 = x1 * 4
    x3 = x2 + 1
    x4 = x3 * 2
    x7 = x1 - x4
    return x7

input = 3.0    
a = Value(input)
actual = fwd(a)
expected = fwd(input)
assert actual.data == expected, f"{actual} vs {expected}"

actual.backward()
grad_fn = jax.grad(fwd)
expected_grad = grad_fn(3.0)
assert_close(a.grad, expected_grad)

