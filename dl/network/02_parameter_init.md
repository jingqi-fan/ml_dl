# 参数初始化

作用
1. 防止梯度消失或爆炸
2. 提高收敛速度
3. 保持对称性破除

常见参数初始化方法
1. 随机初始化
   - 均匀分布初始化 nn.init.uniform_()
     - 默认区间(0, 1), 可以设置为 $(-1/\sqrt{d}, 1/\sqrt{d})$, where d 神经元输入数量
   - 正态分布初始化 nn.init.normal_()
2. 全0初始化 nn.init.zeros_()
3. 全1初始化 ones_()
4. 固定值初始化 constant_()
5. kaiming初始化
   - 均匀分布的kaiming kaiming_uniform_()
     - 从[-limit, limit]中抽取样本，limit = sqrt{6 / fan_in}
   - 正态分布的kaiming kaiming_normal_()
     - 从[0, std]中抽取样本，std = sqrt{2 / fan_in}
   - fan_in 当前层接受的上一层神经元个数
6. xavier初始化
   - 均匀分布的xavier xavier_uniform_()
     - 从[-limit, limit]中抽取样本，$limit = \sqrt{\frac{6}{fan_in+fan_out}}$
   - 正态分布的xavier xavier_normal_()
     - 从[0, std]中抽取样本，$std = \sqrt{\frac{2}{fan_in+fan_out}}$
   - fan_in 当前层接受的上一层神经元个数, fan_out 当前层接受的上一层神经元个数


kaiming和xavier最常用，全0也用
- kaiming+ReLU，
- xavier适用与sigmoid, Tanh，解决梯度消失分体

















