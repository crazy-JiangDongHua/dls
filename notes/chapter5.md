这一张主要介绍了needle库。needle库包括三个py文件：

1. `__init__.py`：将文件夹变为一个Python模块，导入一个模块时，实际上是导入了它的`__init__.py`文件
2. `autograd.py`：包含了五个基本类：
   1. `Value`：表示计算图中的结点的基类
   2. `Op`：表示计算图中的边，即操作基类
   3. `Tensor`：`Value`的子类，多了`grad`成员变量和其他成员函数
   4. `TensorOp`：`Op`的子类，实现了`__call__`来支持调用，返回`Tensor`
3. `ops.py`：包含众多操作类，都是`TensorOp`的子类。



前向传播过程：

1. 调用`Tensor`的运算函数，例如`__add__()`
2. 调用运算符`TensorOp.__call__()`
3. 调用`Tensor.make_from_op()`，该方法首先用`__new__()`创建一个`Tensor`对象，然后用`Value._init()`填上`Tensor`对象的`op`和`input`成员。然后调用`Tensor.realize_cached_data()`计算`cached_dada`成员，也就是实际上的值。

这其中涉及到lazy mode和eager mode两种计算模式，lazy mode下，实际的值要等到被调用时才计算，也就是在前向传播过过程中，不调用`Tensor.realize_cached_data()`方法，等到要用时在调用。eager mode则是直接在前向传播的时候就调用了。



反向传播过程：

1. 得到损失后，调用`bacakward()`
2. 调用`compute_gradient_of_variables()`





Bug：

1. `__rsub__ = __sub__`和`__rmatmul__ = __matmul__`
2. 