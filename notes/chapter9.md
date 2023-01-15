这一章介绍了一些归一化方法和正则化方法



如果没有正确初始化，那么可能导致梯度爆炸或者梯度消失，这样网络根本训练不起来。即使采用了Xavier初始化和kaiming初始化，经过一段时间的训练以后，各个层的输入输出的分布（或者说尺度）依然会变得差距很大。所以在层与层之间添加归一化方法，使得不同层的输出分布变得一致。

1. Layer Normalization和Batch Normalization

   两者都是先归一化均值为0，方差1的分布，然后用 $w$，$b$ 做线性的缩放，即 $x=w\times \frac{x-E[x]}{(Var[x]+\epsilon)^{1/2}}+b$。区别在与两者归一化的维度，参数的个数，以及Batch Normalization在训练和预测时的不同。

   1. BatchNorm是在batch维度做归一化。

      在一维的情况下，输入shape为 (batch_size, feature_size) ，即表格数据，或者为 (batch_size, sequence_length, embedding_size) ，即文本数据，均值的shape（同方差）分别为 (1, feature_size) 和 (1, 1, embedding_size)， $w$，$b$ 的形状和均值相同。

      在二维的情况下，输入shape为 (batch_size, channel_num, H, W) ，即图像数据，均值的shape（同方差）为 (1, channel_num, 1, 1) ， $w$，$b$ 的形状和均值相同。

   2. LayerNorm是在一个example里面做归一化。$w$，$b$ 的形状的形状是归一化掉的维度。

      在一维的情况下，输入shape为 (batch_size, feature_size) ，即表格数据，均值的shape（同方差）为 (batch_size, 1) 。$w$，$b$ 的形状的形状为 (1, feature_size)。

      或者为 (batch_size, sequence_length, embedding_size) ，即文本数据，均值的shape（同方差）可以是 (batch_size, 1, 1)，即一个句子看作一个分布。也可以是 (batch_size, sequence_length, 1)，即一个词看作一个分布，两种在nlp中都有使用，后者可能更多。 $w$，$b$ 的形状分别为 (1, sequence_length, embedding_size) 和 (1, 1, embedding_size)。

      在二维的情况下，输入shape为 (batch_size, channel_num, H, W) ，即图像数据，均值的shape（同方差）为 (batch_size, 1, 1, 1) ， $w$，$b$ 为 (1, channel_num, H, W) 。

   3. BatchNorm在测试的时候，因为可能只有一个一个的输入，求不了均值和方差，所以在训练时，需要求均值和方差的指数移动平均，用作测试的均值和方差。



正则化方法

1. L2正则化：通过优化器的weight_decay就可以实现，注意在求梯度的动量时每一步的梯度需要加上weight_decay*param
2. Dropout