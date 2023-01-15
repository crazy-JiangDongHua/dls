这一张主要介绍了如何实现硬件加速，即如何实现自己的后端。



回顾之前实现的Tensor类，它有一个`cache_data`成员变量，这个变量原来是`numpy.ndarray`，现在我们改成自己实现的`NDArray`类，然后将autograd文件和ops文件中导入的`array_api`模块（原来默认是numpy）改为`NDArray`模块，这样就可以无缝切换了。注意原来使用的numpy的函数，在`NDArray`模块中也要实现哦。

`NDArray`类本质也还只是一个容器，它包含以下成员变量：

- handle: The backend handle that build a flat array which stores the data.
- shape: The shape of the NDArray
- strides: The strides that shows how do we access multi-dimensional elements
- offset: The offset of the first element.
- device: The backend device that backs the computation

其中`handle`是一个自己实现的`Array`类型，是真正存放数据的地方。通过device指定不同的后端，同样，这个可以打包到`Tensor`类中用于指定不同的后端。

然后在不同的后端中，例如numpy、自己写的cpu后端和自己写的gpu后端，实现`Array`类，然后NDArray的各种操作实际上是调用不同后端的`Array`类的函数实现的，因此想要真正加速算子，就是修改`Array`类的算子实现。

自己实现的cpu和gpu后端是c++文件写的，通过`pybind`这个第三方库使得在python中可以调用c++接口。