# Homework 4
Public repository and stub/testing code for Homework 4 of 10-714.



在这个作业里面，由于以前实现的后端代码并不完全兼容numpy接口，但是nn模块里面很多用的又是以前numpy的接口，例如

1. Broadcast_to：nn里面有些模块直接先插入1然后扩展，ndarray或者ops没实现

2. 传入维度里面有很多-1，ndarray和ops也没有实现

所以对hw3中的内容做了修改，增加了更多支持已兼容numpy接口。这些修改应该是在ndarray.py里面完成最好，但是因为一些其他的原因，有些在ops.py里面修改。改动包括：

1. NDArray.reshape支持-1

2. NDArray.broadcast支持自动增加维度，扩张成更大的，同时增加长度检查报错

3. 修改了ndarray_backend_cuda里面的compat函数，将参数strides的类型从uint_32改为int_32，将nadarray_backend_cuda.VecToCuda改为传入vector<int_32>，用来支持负数strides

4. ops.Transpose增加负数轴支持，因为ndarray没有实现swap，且permuter只支持正数输入

5. ops.reshape增加compact调用，因为ndarray.reshape要求不是紧凑的要报错

6. ops.summation增加对多维求和的支持以及负数轴的支持，因为ndarray.sum被写死了

7. ops.logsumexp里面增加了对多维和负数轴的求max，因为ndarray.max被写死了

8. ops.flip增加对多维求和的支持以及负数轴的支持，复用了ops.summation中的代码



还有未完善的：

1. ndarray.getitem只支持正数输入

2. ndarray.matmul不支持批矩阵乘法等等（不支持的太多了，没补了

3. ops.conv只支持stride=1的情况，stride>1存在太多边界条件
