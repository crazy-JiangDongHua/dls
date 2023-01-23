"""Operatpr table."""
# Global operator table.
from math import e
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray

from functools import reduce
import operator

class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


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
        return a + numpy.float32(self.scalar)

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
        return a * numpy.float32(self.scalar)

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


# helper function
def get_pos_axes(axes, ndim):
    '''
    axes is a int , or a list or tuple of axis, maybe include -1，-2
    ndim is the dim number of tensor
    return postive axes
    '''
    axes_list=list(range(ndim))
    if isinstance(axes, int):
        axes=(axes, )
    return list([axes_list[i] for i in axes])


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_axes=list(range(0, a.ndim))
        if self.axes is None:
            new_axes[-2]+=1
            new_axes[-1]-=1
        else:
            # 支持负数轴输入
            pos_axes=get_pos_axes(self.axes, a.ndim)
            new_axes[pos_axes[0]]=pos_axes[1]
            new_axes[pos_axes[1]]=pos_axes[0]
        return a.permute(tuple(new_axes))
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
        return a.compact().reshape(self.shape)
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
        return unbroadcast_to(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION

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

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # 支持负数轴
        if self.axes is None:
            pos_axes=list(range(a.ndim))
        else:
            pos_axes=get_pos_axes(self.axes, a.ndim)
        for i in sorted(pos_axes, reverse=True):
            a=a.sum(axis=i)
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None:
            newshape=list(node.inputs[0].shape)
            if isinstance(self.axes, int):
                newshape[self.axes]=1
            else:
                for i in self.axes:
                    newshape[i]=1
            out_grad=reshape(out_grad, tuple(newshape))
        return broadcast_to(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
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


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mask = node.inputs[0].realize_cached_data()>0.0
        return out_grad*Tensor(mask, device=node.device,dtype=node.dtype)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # 支持多维度轴和负数轴
        if self.axes is None:
            pos_axes=list(range(Z.ndim))
        else:
            pos_axes=get_pos_axes(self.axes, Z.ndim)
        Z_max=Z+0.0
        for i in sorted(pos_axes, reverse=True):
            Z_max=Z_max.max(axis=i, keepdims=True)
        Z_exp=array_api.exp(Z-Z_max.broadcast_to(Z.shape))
        Z_sum=Z_exp+0.0
        for i in sorted(pos_axes, reverse=True):
            Z_sum=Z_sum.sum(axis=i, keepdims=False)
        return array_api.log(Z_sum)+Z_max.reshape(Z_sum.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z=node.inputs[0]
        if self.axes is not None:
            newshape=list(Z.shape)
            if isinstance(self.axes, int):
                newshape[self.axes]=1
            else:
                for i in self.axes:
                    newshape[i]=1
            node=reshape(node, tuple(newshape))
            out_grad=reshape(out_grad, tuple(newshape))
        node=broadcast_to(node, Z.shape)
        out_grad=broadcast_to(out_grad, Z.shape)
        return exp(Z-node)*out_grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (1-node**2)*out_grad
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        shape=args[0].shape
        for arg in args:
            assert shape==arg.shape
        out_shape=list(shape)
        out_shape.insert(self.axis, len(args))
        slices=list([slice(s) for s in out_shape])
        out=array_api.empty(tuple(out_shape), dtype=args[0].dtype, device=args[0].device)
        # new_shape=list(shape)
        # new_shape.insert(self.axis, 1)
        for i, arg in enumerate(args):
            slices[self.axis]=i
            out[tuple(slices)]=arg
        return out
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        out_shape=list(A.shape)
        del out_shape[self.axis]
        out_shape=tuple(out_shape)
        slices=list([slice(s) for s in A.shape])
        out=[]
        for i in range(A.shape[self.axis]):
            slices[self.axis]=i
            out.append(A[tuple(slices)].compact().reshape(out_shape))
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # 支持负数轴
        if self.axes is None:
            pos_axes=list(range(a.ndim))
        else:
            pos_axes=get_pos_axes(self.axes, a.ndim)
        return a.flip(tuple(pos_axes)) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            pos_axes=list(range(node.ndim))
        else:
            pos_axes=get_pos_axes(self.axes, len(node.shape))
        return flip(out_grad, tuple(pos_axes))
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # 支持负数轴
        if self.axes is None:
            pos_axes=list(range(a.ndim))
        else:
            pos_axes=get_pos_axes(self.axes, a.ndim)
        new_shape=list(a.shape)
        for axis in pos_axes:
            new_shape[axis]*=(self.dilation+1)
        out=array_api.full(tuple(new_shape), 0.0, dtype=a.dtype, device=a.device)
        slices=tuple([slice(0, s, self.dilation+1) \
                      if i in pos_axes else slice(s) \
                      for i, s in enumerate(new_shape)])
        out[slices]=a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            pos_axes=list(range(a.ndim))
        else:
            pos_axes=get_pos_axes(self.axes, a.ndim)
        slices=tuple([slice(0, s, self.dilation+1) \
                      if i in pos_axes else slice(s) \
                      for i, s in enumerate(a.shape)])
        return a[slices]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilate)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


# may appear bug when stride>1
class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, img, weight):
        ### BEGIN YOUR SOLUTION
        assert img.ndim==4 and weight.ndim==4
        img=img.pad(((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        N,H,W,C_in = img.shape
        K,_,_,C_out = weight.shape
        Ns, Hs, Ws, Cs = img.strides
        out_H=1+(H-K)//self.stride
        out_W=1+(W-K)//self.stride
        img2col=img.as_strided((N,out_H,out_W,K,K,C_in),\
                             (Ns,Hs*self.stride,Ws*self.stride,Hs,Ws,Cs))\
                             .compact()\
                             .reshape((N*out_H*out_W,-1))
        weight=weight.compact().reshape((-1,C_out))
        return (img2col @ weight).compact().reshape((N,out_H,out_W,C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        img, weight=node.inputs
        N,H,W,C_in = img.shape
        K,_,_,C_out = weight.shape
        
        # 扩张out_grad
        if self.stride>1:
            out_grad=dilate(out_grad, (1,2), self.stride-1)
        
        # 求img_grad
        # 左右和上下翻转 
        w_f=flip(weight,(0,1))
        # w_p: K*K*C_out*C_in
        w_p=transpose(w_f,(2,3))
        img_grad=conv(out_grad, w_p, padding=K-1-self.padding)

        # 求w_grad
        # img_p: C_in*H*W*N
        img_p=transpose(img, (0,3))
        # grad_p: out_H*out_W*N*C_out
        grad_p=transpose(transpose(out_grad, (0,1)), (1,2))
        # w_grad: C_in*K*K*C_out
        w_grad=conv(img_p, grad_p, padding=self.padding)
        # w_grad: K*K*C_in*C_out
        w_grad=transpose(transpose(w_grad, (0,1)), (1,2))

        return img_grad, w_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



