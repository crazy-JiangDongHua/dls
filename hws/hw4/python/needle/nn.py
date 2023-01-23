"""The module.
"""
from typing import List
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
        self.weight=Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True, device=device, dtype=dtype))
        if bias:
            # 注意reshape不能放在Parameter()之后，因为这样self.bias就不再是叶节点
            # 在pytorch中非叶节点不能求梯度，而在needle中这样做不影响求梯度，但是会让bias变成Tensor类
            # 这里bias的形状测试与ipynb里面描述的有出入
            self.bias=Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True, device=device, dtype=dtype).reshape((1, out_features)))
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
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1+ops.exp(-x))**-1
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
    def __getitem__(self, idx):
        return self.modules[idx]


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        exp_sum=ops.logsumexp(logits, (-1,)).sum()
        z_y_sum=(init.one_hot(logits.shape[-1], y, requires_grad=True, device=logits.device)*logits).sum()
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
        self.weight=Parameter(init.ones(dim, requires_grad=True, device=device, dtype=dtype))
        self.bias=Parameter(init.zeros(dim, requires_grad=True, device=device, dtype=dtype))
        # pytorch里面是用buffer表示不可学习的参数
        self.running_mean=init.zeros(dim, requires_grad=False, device=device, dtype=dtype)
        self.running_var=init.ones(dim, requires_grad=False, device=device, dtype=dtype)
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


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


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




class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.kaiming_uniform(
            in_channels*kernel_size*kernel_size,
            out_channels*kernel_size*kernel_size,
            shape=(kernel_size, kernel_size, in_channels, out_channels),
            device=device,
            dtype=dtype,
            requires_grad=True
        ))
        if bias:
            bias_bound = 1/(in_channels*kernel_size**2)**0.5
            self.bias=Parameter(init.rand(
                out_channels,
                low=-bias_bound,
                high=bias_bound,
                device=device,
                dtype=dtype,
                requires_grad=True
            ))
        else:
            self.bias=None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x=x.transpose((1,2)).transpose((2,3))
        same_padding=(self.kernel_size-1)//2
        out=ops.conv(x, self.weight, stride=self.stride, padding=same_padding)
        if self.bias is not None:
            out+=self.bias.broadcast_to(out.shape)
        out=out.transpose((2,3)).transpose((1,2))
        return out
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size=hidden_size
        self.device = device
        self.dtype = dtype
        sqrtk=(1/hidden_size)**0.5
        self.W_ih=Parameter(init.rand(input_size, hidden_size, low=-sqrtk, high=sqrtk,
                          requires_grad=True, device=device, dtype=dtype))
        self.W_hh=Parameter(init.rand(hidden_size, hidden_size, low=-sqrtk, high=sqrtk,
                          requires_grad=True, device=device, dtype=dtype))
        if bias:
            self.bias_ih=Parameter(init.rand(hidden_size, low=-sqrtk, high=sqrtk,
                          requires_grad=True, device=device, dtype=dtype))
            self.bias_hh=Parameter(init.rand(hidden_size, low=-sqrtk, high=sqrtk,
                          requires_grad=True, device=device, dtype=dtype))
        else:
            self.bias_ih=None
            self.bias_hh=None
        if nonlinearity=='tanh':
            self.nonlinearity=Tanh()
        elif nonlinearity=='relu':
            self.nonlinearity=ReLU()
        else:
            raise ValueError 
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        out=X@ self.W_ih
        batch_size, _=X.shape
        if h is None:
            h = init.zeros(batch_size, self.hidden_size, device=self.device, 
                           dtype=self.dtype, requires_grad=True)
        out+=h@ self.W_hh
        # else h==0
        if self.bias_ih is not None:
            out+=self.bias_ih.broadcast_to(out.shape)
        if self.bias_hh is not None:
            out+=self.bias_hh.broadcast_to(out.shape)
        return self.nonlinearity(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size=hidden_size
        self.device = device
        self.dtype = dtype
        self.num_layers=num_layers
        self.rnn_cells=[RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity, 
                            device=device, dtype=dtype)] +\
                       [RNNCell(hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity, 
                            device=device, dtype=dtype) for _ in range(num_layers-1)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        X_list=list(ops.split(X, 0))
        _, batch_size, _=X.shape
        h_list=list(ops.split(h0, 0)) \
               if h0 is not None else \
               list([init.zeros(batch_size, self.hidden_size, device=self.device, 
                  dtype=self.dtype) for _ in range(self.num_layers)])
        for l,rnn in enumerate(self.rnn_cells):
            h=h_list[l]
            for t,x in enumerate(X_list):
                h=rnn(x,h)
                X_list[t]=h
            h_list[l]=h
        return ops.stack(X_list,0), ops.stack(h_list, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size=hidden_size
        self.device = device
        self.dtype = dtype
        sqrtk=(1/hidden_size)**0.5
        self.W_ih=Parameter(init.rand(input_size, 4*hidden_size, low=-sqrtk, high=sqrtk,
                          requires_grad=True, device=device, dtype=dtype))
        self.W_hh=Parameter(init.rand(hidden_size, 4*hidden_size, low=-sqrtk, high=sqrtk,
                          requires_grad=True, device=device, dtype=dtype))
        if bias:
            self.bias_ih=Parameter(init.rand(4*hidden_size, low=-sqrtk, high=sqrtk,
                          requires_grad=True, device=device, dtype=dtype))
            self.bias_hh=Parameter(init.rand(4*hidden_size, low=-sqrtk, high=sqrtk,
                          requires_grad=True, device=device, dtype=dtype))
        else:
            self.bias_ih=None
            self.bias_hh=None
        self.sigmoid=Sigmoid()
        self.tanh=Tanh()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        out=X@ self.W_ih
        if h is not None:
            h0, c0=h
        else:
            batch_size,_=X.shape
            h0=init.zeros(batch_size, self.hidden_size, device=self.device, 
                          dtype=self.dtype, requires_grad=True)
            c0=init.zeros(batch_size, self.hidden_size, device=self.device, 
                          dtype=self.dtype, requires_grad=True)
        out+=h0@ self.W_hh
        # else h==0
        if self.bias_ih is not None:
            out+=self.bias_ih.broadcast_to(out.shape)
        if self.bias_hh is not None:
            out+=self.bias_hh.broadcast_to(out.shape)
        bs, hs_4=out.shape
        i,f,g,o=ops.split(ops.transpose(out.reshape((bs, 4, -1))), 2)
        i=self.sigmoid(i)
        f=self.sigmoid(f)
        g=self.tanh(g)
        o=self.sigmoid(o)
        c_=f*c0+i*g
        h_=o*self.tanh(c_)
        return h_, c_
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.hidden_size=hidden_size
        self.device = device
        self.dtype = dtype
        self.num_layers=num_layers
        self.lstm_cells=[LSTMCell(input_size, hidden_size, bias=bias, 
                              device=device, dtype=dtype)] +\
                        [LSTMCell(hidden_size, hidden_size, bias=bias,
                              device=device, dtype=dtype) for _ in range(num_layers-1)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        X_list=list(ops.split(X, 0))
        _, batch_size, _=X.shape
        h_list=[list(ops.split(h[0], 0)), list(ops.split(h[1], 0))] \
                if h is not None else \
               [list([init.zeros(batch_size, self.hidden_size, device=self.device, 
                  dtype=self.dtype) for _ in range(self.num_layers)]),
                list([init.zeros(batch_size, self.hidden_size, device=self.device, 
                  dtype=self.dtype) for _ in range(self.num_layers)])]
        for l,rnn in enumerate(self.lstm_cells):
            h,c=h_list[0][l], h_list[1][l]
            for t,x in enumerate(X_list):
                h,c=rnn(x,(h,c))
                X_list[t]=h
            h_list[0][l], h_list[1][l]=h,c
        return ops.stack(X_list,0), (ops.stack(h_list[0], 0), ops.stack(h_list[1], 0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.rand(num_embeddings, embedding_dim, requires_grad=True, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        t, b=x.shape
        x_one_hot=init.one_hot(self.weight.shape[0], x.reshape(-1), requires_grad=True, device=x.device, dtype=x.dtype)
        return (x_one_hot@ self.weight).reshape((t, b, -1))
        ### END YOUR SOLUTION
