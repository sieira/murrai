import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        out = Value(-self.data, (self,), "-")

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("exponent must be a plain number")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        if self.data <= 0:
            raise ValueError("log needs a positive number")
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def zero_grad(self):
        self.grad = 0.0

    def __repr__(self):
        return f"Value({self.data}, grad={self.grad})"

    def backward(self):
        order = []
        seen = set()

        def build(node):
            if node in seen:
                return
            seen.add(node)
            for child in node._prev:
                build(child)
            order.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(order):
            node._backward()


def finite_difference_grads(build, leaves, step=1e-6):
    grads = []
    for leaf in leaves:
        original = leaf.data
        leaf.data = original + step
        up = build(*leaves).data
        leaf.data = original - step
        down = build(*leaves).data
        leaf.data = original
        grads.append((up - down) / (2 * step))
    return grads
