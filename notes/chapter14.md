这一张主要介绍卷积操作的实现。



首先定义一下图像和卷积核的形式。

* 图像：`float Z[BATCHES][HEIGHT][WIDTH][CHANNELS]`

* 卷积核：`float weights[KERNEL_SIZE][KERNEL_SIZE][IN_CHANNELS][OUT_CHANNELS]`

内存排列也是行优先的，也就是连续的（或者说是紧凑的）。

这里的格式和pytorch有所不同，pytorch是

* 图像：`float Z[BATCHES][CHANNELS][HEIGHT][WIDTH]`

* 卷积核：`float weights[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE]`

但是在内存排列上，图像的内存排列是和前面本文采用格式一致的。

这里先假设没有padding和stride

## 1. 全 for-loop 实现

```python
def conv_naive(Z, weight):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    
    out = np.zeros((N,H-K+1,W-K+1,C_out));
    for n in range(N):
        for c_in in range(C_in):
            for c_out in range(C_out):
                for y in range(H-K+1):
                    for x in range(W-K+1):
                        for i in range(K):
                            for j in range(K):
                                out[n,y,x,c_out] += Z[n,y+i,x+j,c_in] * weight[i,j,c_in,c_out]
    return out
```

这样的实现肯定是非常慢的，我们的目标是消除循环，利用矩阵乘法计算。

## 2. 矩阵乘法实现

举一个方便理解的例子，假设卷积核只有 $1\times 1$ 大小，可以把图像看作由众多单像素位置全输入通道的向量组成，卷积核可以看作`weight[IN_CHANNELS][OUT_CHANNELS]`的矩阵，那么卷积操作可以看作是图像与卷积核的批矩阵乘法。

由此对于一般的卷积核，我们只需要循环卷积核的两维，卷积实现如下：

```python
def conv_matrix_mult(Z, weight):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    out = np.zeros((N,H-K+1,W-K+1,C_out))
    
    for i in range(K):
        for j in range(K):
            out += Z[:,i:i+H-K+1,j:j+W-K+1,:] @ weight[i,j]
    return out
```

## 3. img2col实现

如之前的笔记所述，可以将图像展开为矩阵，把卷积核展开为向量，然后作矩阵乘法实现卷积。

先举一个单样本单输入通道单输出通道的例子。

输入图像Z如下：

```python
import numpy as np
n = 6
A = np.arange(n**2, dtype=np.float32).reshape(n,n)
print(A)
'''
[[ 0.  1.  2.  3.  4.  5.]
 [ 6.  7.  8.  9. 10. 11.]
 [12. 13. 14. 15. 16. 17.]
 [18. 19. 20. 21. 22. 23.]
 [24. 25. 26. 27. 28. 29.]
 [30. 31. 32. 33. 34. 35.]]
'''
```

有一种很bug的方法可以查看A（numpy.ndarray）的底层数据排列，

```python
import ctypes
print(np.frombuffer(ctypes.string_at(A.ctypes.data, A.nbytes), dtype=A.dtype, count=A.size))
'''
[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.]
'''
```

可以看出底层的数据都是用一维数组存储的，使用用不同的shape和strides提供的不同的view（这个词我不知道怎么翻译比较好）。

此时A的strides是(24, 4)，是连续（或者说紧凑）排列的。

```python
print(A.strides)
'''
(24, 4)
'''
```

假设我们的卷积核是 $3\times 3$的。

```python
weight = np.arange(9, dtype=np.float32).reshape(3,3)
```

那么最后卷积后得到的图像是$(H-K+1)\times(W-K+1)=4\times 4$的，所以img2col需要将图像转化为$(H-K+1)\times(W-K+1)\times K\times K=4\times 4\times 3\times 3$的张量，每一个$3\times 3$的矩阵与卷积核元素乘以后求和，就称为了输出矩阵的一个元素。

所以我们期望得到的img2col矩阵是

```python
'''
[[[[ 0.  1.  2.]
   [ 6.  7.  8.]
   [12. 13. 14.]]

  [[ 1.  2.  3.]
   [ 7.  8.  9.]
   [13. 14. 15.]]

  [[ 2.  3.  4.]
   [ 8.  9. 10.]
   [14. 15. 16.]]

  [[ 3.  4.  5.]
   [ 9. 10. 11.]
   [15. 16. 17.]]]


 [[[ 6.  7.  8.]
   [12. 13. 14.]
   [18. 19. 20.]]

  [[ 7.  8.  9.]
   [13. 14. 15.]
   [19. 20. 21.]]

  [[ 8.  9. 10.]
   [14. 15. 16.]
   [20. 21. 22.]]

  [[ 9. 10. 11.]
   [15. 16. 17.]
   [21. 22. 23.]]]


 [[[12. 13. 14.]
   [18. 19. 20.]
   [24. 25. 26.]]

  [[13. 14. 15.]
   [19. 20. 21.]
   [25. 26. 27.]]

  [[14. 15. 16.]
   [20. 21. 22.]
   [26. 27. 28.]]

  [[15. 16. 17.]
   [21. 22. 23.]
   [27. 28. 29.]]]


 [[[18. 19. 20.]
   [24. 25. 26.]
   [30. 31. 32.]]

  [[19. 20. 21.]
   [25. 26. 27.]
   [31. 32. 33.]]

  [[20. 21. 22.]
   [26. 27. 28.]
   [32. 33. 34.]]

  [[21. 22. 23.]
   [27. 28. 29.]
   [33. 34. 35.]]]]
'''
```

那么要怎么做呢，一个一个复制吗，nono，只需要调整shape和strides就可以了，这样就能够在不改变底层数据的条件下，获得新的张量。假设转换后的张量是B，那么B的shape就是(4,4,3,3)，strides可以通过计算得知是(6,1,6,1)，很巧合的是，A的stride就是(6,1)（numpy显示(24,6)是乘了byte值，float类型占了4byte），实际上img2col在转换时，只需要把H和W两维扩张成四维，然后strides就是(Hs,Ws,Hs,Ws)。

```python
B = np.lib.stride_tricks.as_strided(A, shape=(4,4,3,3), strides=4*(np.array((6,1,6,1))))
```

查看B的底层，果然没变

```python
print(np.frombuffer(ctypes.string_at(B.ctypes.data, size=A.nbytes), B.dtype, A.size))
'''
[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.]
'''
```

最后就是$3\times 3$的矩阵与卷积核元素乘以后求和的过程可以通过向量乘法实现

```python
C = B.reshape(16,9)
(C @ W.reshape(9)).reshape(4,4)
'''
array([[ 366.,  402.,  438.,  474.],
       [ 582.,  618.,  654.,  690.],
       [ 798.,  834.,  870.,  906.],
       [1014., 1050., 1086., 1122.]], dtype=float32)
'''
```

有一点需要注意的是，B调用reshape生成C的时候，reshape其实隐式的调用了ascontinuousarray，这个函数会改变底层内存，因为B是不紧凑的，所以生成C变得紧凑了，也就是产生了实际的复制过程，或者说生成img2col矩阵的过程。打印C的strides看看

```python
C.strides
'''
(36, 4)
'''
```

直接生成img2col矩阵这种方式效率是不高的，而且会占用巨大的内存，现在主流的方式有两种，一种是采用lazy模式，需要的时候再生成对应的部分。一种不生成矩阵了，直接用更原生的指令遍历。

拓展到多样本多输入通道多输出通道就很简单了。输出是4维张量 $(H-K+1)\times(W-K+1)\times K\times K$，img2col矩阵是6维张量 $N\times (H-K+1)\times(W-K+1)\times K\times K\times C_{IN}$，然后展平img2col矩阵的后三维，和卷积核的前三维，然后作矩阵乘法就可以了，即$(N\times (H-K+1)\times(W-K+1))\times (K\times K\times C_{IN})$ 乘上 $(K\times K\times C)\times C_{out}$。最后实现如下。

```python
def conv_im2col(Z, weight):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    Ns, Hs, Ws, Cs = Z.strides
    
    inner_dim = K * K * C_in
    A = np.lib.stride_tricks.as_strided(Z, shape = (N, H-K+1, W-K+1, K, K, C_in),
                                        strides = (Ns, Hs, Ws, Hs, Ws, Cs)).reshape(-1,inner_dim)
    out = A @ weight.reshape(-1, C_out)
    return out.reshape(N,H-K+1,W-K+1,C_out)
```



还有两个问题，等实现的时候再回来想：

1. 实现padding和stride

   * padding可以简单通过申请一块新的内存（全部填0），然后将原图复制到新内存中
   * stride可以通过修改strides实现，如果stride是4，那么将img2col矩阵的strides改成`(Ns, 4*Hs, 4*Ws, Hs, Ws, Cs)`就可以生成新的矩阵

2. 反向传播：

   以下用conv(Z, weight)代表self，out=conv(Z, weight)

   1. Z.grad：Z.grad可以通过将weight翻转（上下和左右），然后对out.grad做卷积得到，其中stride=1。self.stride=1的时候padding=K-1-self.padding，当self.stride>1时，会出现左右padding不一致以及padding为负数的边界情况，暂时不想了。
   2. weight.grad：weight.grad可以通过out.grad对Z做卷积得到，其中需要把批量维度B作为输入通道实现梯度求和，其中stride=1。self.stride=1的时候padding=self.padding，当self.stride>1时，会出现好多边界情况，暂时不想了。



对于卷积层中还会存在的偏置b，它的梯度就是out.grad求和就行了，输出维度为(Cout,)。

