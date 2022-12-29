这一张主要介绍自动微分系统的设计。



微分方法包括：

1. 数值微分法：

   通过 $\frac{\partial f(\theta)}{\partial theta_i}=\lim_{\epsilon \rightarrow 0} \frac{f(\theta + \epsilon e_i)-f(\theta)}{\epsilon}$ 或者 $\frac{\partial f(\theta)}{\partial theta_i}=\lim_{\epsilon \rightarrow 0} \frac{f(\theta + \epsilon e_i)-f(\theta - \epsilon e_i)}{2\epsilon}$ 求微分，其中 $e_i$ 是和 $\theta$ 相同维度的向量，在 $\theta_i$ 的位置为1，其他的位置都为0。两个公式的误差项分别为 $O(\epsilon)$ 和 $O(\epsilon^2)$ 。

   数值微分计算简单，但是需要遍历所有参数，所以计算效率很低，而且要求 $\epsilon$ 很小，很容易导致数值误差，

   所以数值微分一般用于检查自动微分方法是否计算正确，但是也不是一个一个参数去检查，而是采用公式 $\delta^T \nabla f(\theta)=\lim_{\epsilon \rightarrow 0} \frac{f(\theta + \epsilon e_i)-f(\theta - \epsilon e_i)}{2\epsilon}$ ，其中 $\delta$ 是从单位球中随机采样的向量，通过检查公式两端误差可以判断自动微分系统是否计算正确。

2. 手动微分法

   对于深度学习而言，因为模型大，参数多等原因，手动计算梯度几乎不可能。

3. 自动微分

   计算可以表示为一个有向无环图，其中结点表示中间结果，边表示计算操作，这种图交有向无环图

   自动微分可以分为前向模型自动微分（Forward mode automatic differentiation）和反向模型自动微分（Reverse mode automatic differentiation）。前向模型运行次数正比于输入参数的数量，适用于输入参数少，输出结果多的情况。反向模型运行次数正比于输出结果的数量，适用于输入参数多，输出结果少的情况，而大多数深度学习都是这种，输入参数好几百亿，输出结果只有一个损失函数值。

   反向传播中，多路径的微分可以直接通过各个路径的梯度相加实现。

   第一代的反向传播是在原本的计算图上反向计算实现的。第二代的反向传播把反向计算的过程变成了一个拓展计算图，这样前向传播和反向传播可以表示为一张合并的计算图，这么做的好处有两个：1. 可以方便优化；2. 可以方便计算二阶导数乃至更高阶导数（继续补上计算图）。