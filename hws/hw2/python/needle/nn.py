"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True))
        if bias:
            # 注意reshape不能放在Parameter()之后，因为这样self.bias就不再是叶节点
            # 在pytorch中非叶节点不能求梯度，而在needle中这样做不影响求梯度，但是会让bias变成Tensor类
            # 这里bias的形状测试与ipynb里面描述的有出入
            self.bias=Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True).reshape((1, out_features)))
        else:
            self.bias=None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X=X@ self.weight
        if self.bias is not None:
            X+=ops.broadcast_to(self.bias, X.shape)
        return X
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        from operator import mul
        from functools import reduce
        dim=reduce(mul, X.shape[1:])
        return X.reshape((X.shape[0], dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x=module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        exp_sum=ops.logsumexp(logits, (-1,)).sum()
        z_y_sum=(init.one_hot(logits.shape[-1], y, requires_grad=True)*logits).sum()
        # numpy中 float32/int64 可能自动类型提升为float64 
        return (exp_sum-z_y_sum)/logits.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.ones(dim, requires_grad=True))
        self.bias=Parameter(init.zeros(dim, requires_grad=True))
        # pytorch里面是用buffer表示不可学习的参数
        self.running_mean=init.zeros(dim, requires_grad=False)
        self.running_var=init.ones(dim, requires_grad=False)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            batch_size=x.shape[0]
            x_mean=x.sum((0,))/batch_size
            x_var=((x-x_mean.broadcast_to(x.shape))**2).sum((0,))/batch_size
            # 不产生计算图
            # 这和常用的指数引动平均有点不一样 一般是 running_mean=beta*running_mean+(1-beta)*x_mean
            self.running_mean.data=(1-self.momentum)*self.running_mean+self.momentum*x_mean.data
            self.running_var.data=(1-self.momentum)*self.running_var+self.momentum*x_var.data
            # 训练时仍然使用小批量数据的均值和方差
            x_norm=(x-ops.broadcast_to(x_mean, x.shape))/((ops.broadcast_to(x_var, x.shape)+self.eps)**0.5)
        else:
            x_norm=(x-ops.broadcast_to(self.running_mean, x.shape))/((ops.broadcast_to(self.running_var, x.shape)+self.eps)**0.5)
        return x_norm*ops.broadcast_to(self.weight, x.shape)+ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.ones(dim, requires_grad=True))
        self.bias=Parameter(init.zeros(dim, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feature_size=x.shape
        # 这里需要先reshape一下，(m,n)经过求和转为(m,)，(m,)不能直接广播为(m,n)
        x_mean=x.sum((1,)).reshape((batch_size, 1))/feature_size
        x_var=((x-ops.broadcast_to(x_mean, x.shape))**2).sum((1,)).reshape((batch_size, 1))/feature_size
        x_norm=(x-ops.broadcast_to(x_mean, x.shape))/((ops.broadcast_to(x_var, x.shape)+self.eps)**0.5)
        return x_norm*ops.broadcast_to(self.weight, x.shape)+ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask=init.randb(*x.shape, p=1-self.p, dtype="float32")
            return x*mask/(1-self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION



