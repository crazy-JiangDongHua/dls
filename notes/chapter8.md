本章介绍了一些帮助实现高级接口的内容



1. `Tensor.data`和`Tensor.detach()`返回`Tensor `从计算图中摘出的版本，并和原`Tensor`共享底层数据`Tensor.cache_data`。这实际上是通过调用`Tensor.make_const()`方法，将`Tensor.op`和`Tensor.inputs`分别设置为`None`和`[]`实现的。同时在eager模式下，如果`Tensor.require_grad=False`将默认返回`Tensor.data`。

   `Tensor.data`也被设置了`setter`属性，赋值可以直接修改`cache_data`

   可以用来实现优化器

2. softmax的数值稳定性，令 $z = softmax(x)$ ，则有 $z_i = \frac{exp(x_i)}{\sum_k exp(x_k)}$ ，但是当 $x_i$ 很大时候，$exp(x_i)$ 将趋向于正无穷，导致上溢出。一个简单的技巧就是：$z_i = \frac{exp(x_i-max(x_i))}{\sum_k exp(x_k-max(x_i))}$，这样就可以避免分子上溢出，同时分母至少有一个值为1，也不会下溢出。但是接下来计算交叉熵时，也有可能发生下溢出。当 $x_i$ 为负数，且绝对值很大时，此时分母是一个很小的正数，导致 $z_i = \frac{exp(x_i-max(x_i))}{\sum_k exp(x_k-max(x_i))}=0$ 发生下溢出，接着计算 $\log z_i=\log 0=-\infty$，这就因为舍入误差导计算错误，所以可以变换计算形式：$\log z_i=\log \frac{exp(x_i-max(x_i))}{\sum_k exp(x_k-max(x_i))}=x_i-max(x_i))-\log \sum_k exp(x_k-max(x_i))$。

3. 定义`Praameter`类，就是一种特殊的`Tensor`

   ```python
   class Parameter(ndl.Tensor):
       """parameter"""
   ```

   

4. 定义`Module`类：

   ```python
   def _get_params(value):
       if isinstance(value, Parameter):
           return [value]
       if isinstance(value, dict):
           params = []
           for k, v in value.items():
               params += _get_params(v)
           return params
       if isinstance(value, Module):
           return value.parameters()
       return []
   
   class Module:
       def parameters(self):
           return _get_params(self.__dict__)
   
       def __call__(self, *args, **kwargs):
           return self.forward(*args, **kwargs)
   ```

   

5.  所有的Loss Funtion都可以看作特殊的`Module`

6. 定义`Optimizer`类

   ```python
   class Optimizer:
       def __init__(self, params):
           self.params = params
   
       def reset_grad(self):
           for p in self.params:
               p.grad = None
           
       def step(self):
           raise NotImplemented()
           
   class SGD(Optimizer):
       def __init__(self, params, lr):
           super()__init__(params)
           self.lr = lr
   
       def step(self):
           for w in self.params:
               w.data = w.data + (-self.lr) * w.grad
   ```

   

7. 初始化方法，介绍了kaiming初始化

8. 实现`TupleTensor`