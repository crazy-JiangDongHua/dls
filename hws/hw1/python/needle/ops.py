"""Operator implementations."""

from numbers import Number
from typing import Optional, List

from IPython.utils.path import unescape_glob
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return self.scalar*(node.inputs[0]**(self.scalar-1))*out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b=node.inputs
        return out_grad/b, -out_grad*a/(b**2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None:
            return array_api.swapaxes(a, *self.axes)
        return array_api.swapaxes(a, -1, -2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 包含了被复制的轴，也就是要被求和的轴
        return unbroadcast_to(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

# helpfun 逆boradcast_to操作
def unbroadcast_to(a, inshape):
    # 包含了被复制的轴，也就是要被求和的轴
    axis=[]
    padnum=0
    # 左侧插入的轴需要被求和
    if len(a.shape)>len(inshape):
        padnum=len(a.shape)-len(inshape)
        axis+=list(range(padnum))
    for i in range(len(inshape)):
        # 也有可能a.shape[i+padnum]=1，但是没关系，最后的reshape会恢复
        if inshape[i]==1:
            axis.append(i+padnum)
    if len(axis)!=0:
        a=summation(a, tuple(axis))
    return reshape(a, inshape)



class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None:
            newshape=list(node.inputs[0].shape)
            # 有一个bug，如果输入axes是int，可以前向传播，反向这里就会报错了
            for i in self.axes:
                newshape[i]=1
            out_grad=reshape(out_grad, tuple(newshape))
        return broadcast_to(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)

# np.matmul定义如下
# 计算两个数组的矩阵积：
# 如果两者都是2-D数组，此时就像我们常见的矩阵乘法
# 如果任意一个参数的维度是N-D(N > 2)，它将被视为位于最后两个维度中的矩阵的堆叠，并相应地广播。
# 如果a的维度是1-D，它会通过在左边插入1到它的维度提升为矩阵，然后与b进行矩阵乘法，完了之后插入的1会被移除
# 如果b的维度是1-D，它会通过在右边插入1到它的维度提升为矩阵，然后与a进行矩阵乘法，完了之后插入的1会被移除
class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a@b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b=node.inputs
        grad_a, grad_b=matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad)
        # unbroadcast_to
        return unbroadcast_to(grad_a, a.shape), unbroadcast_to(grad_b, b.shape)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mask = node.inputs[0].realize_cached_data()>0.0
        return out_grad*Tensor(mask, dtype=array_api.float32)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

