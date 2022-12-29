这一章介绍了全联接神经网络，一些优化算法和初始化方法。



全联接神经网络介绍了定义。



优化算法介绍了：

1. 批梯度下降

2. 牛顿法

   深度学习用不了牛顿法的原因在于二阶导的逆难以计算（$O(n^3)$），甚至存储都是大问题（$O(n^2)$）。所以更好的方法是牛顿法的近似。

3. 动量法

   有两种写法：

   1. 采用指数移动平均：$u_{t+1}=\beta u_t+(1-\beta)\nabla_\theta f(\theta_t),\quad \theta_{t+1}=\theta_t-\alpha u_{t+1}$。
   2. 直接加，衰减过去梯度：$u_{t+1}=\beta u_t+\nabla_\theta f(\theta_t),\quad \theta_{t+1}=\theta_t-\alpha u_{t+1}$。

   采用移动指数平均可以使 $u$ 的尺度和梯度一致。但是指数移动平均的预热期（一般 $u_0=0$ ）， $u_t$ 会比之后的尺度小很多，所以一般加一个消除这种偏差的项，即 $\theta_{t+1}=\theta_t-\alpha u_{t+1}/(1-\beta^{t+1})$，使得初期的 $u_t$ 和之后的尺度相同。

4. Nesterov momentum

   求梯度之前，先“向前看”一步。$\theta_t^\prime=\theta_t-\alpha\beta u_t,\quad u_{t+1}=\beta u_t+(1-\beta)\nabla_{\theta_t^\prime} f(\theta_t^\prime),\quad \theta_{t+1}=\theta_t-\alpha u_{t+1}$

   为了和其他的梯度下降采用相同的形式，避免在前向传播之前还要修改 $\theta_t^\prime$ ，一般在实现过程中我们直接维护$\theta_t^\prime$ ，每次实现如下：

   ```python
   u_prev = u
   u = beta*u + (1-beta)*grad
   theta = theta-alpha*(1+beta)*u+alpha*beta+u_prev
   ```

5. Adam

6. 随机梯度下降



初始化方法介绍了

1. 参数不能全初始化为0，要么梯度全为0，要么每一层的权重都一样，隐藏层单元相当于一个。

2. 随机初始化，0均值，但是需要正确的方差，目标是使得输入输出方差一致

3. Xavier初始化

   假设输入 $x, w\in R^n,\ x_i\sim N(0,1),\ w_i\sim N(0,\frac{1}{n})$，那么 $E[x_iw_i]=E[x_i]E[w_i]=0$，$Var[x_iw_i]=Var[x_i]Var[w_i]=\frac{1}{n}$。所以有$E[w^Tx]=0$，$Var[w^Tx]=1$。假设激活函数是线性的，那么就有 $z_i\sim N(0,1)$，这样层的输入和输出分布就是一致的。当考虑前向传播的时候（fan-in），$w_i\sim N(0,\frac{1}{n_{in}})$。当考虑反向传播的时候（fan-out），$w_i\sim N(0,\frac{1}{n_{out}})$。同时考虑的时候，$w_i\sim N(0,\frac{2}{n_{in}+n_{out}})$。

   Xavier初始化适合于tanh作为激活函数的时候，因为近似线性且关于原点对称。

4. Kaiming初始化

   当激活函数不再是线性的，且不再关于原点对称，例如relu。上述假设就不成立了。对于relu而言，因为 $w^tx\sim N(0,1)$ ，所以有一半的元素被置零，所以方差减半，所以当考虑前向传播的时候（fan-in），$w_i\sim N(0,\frac{2}{n_{in}})$，其他类似。