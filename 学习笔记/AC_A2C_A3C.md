# AC、A2C、A3C



## 一、Policy Gradient 算法

​		在PG算法中，我们的目的是求得策略函数$P(a|s;\theta)$使得目标函数最大。对于马尔科夫，一个序列τ发生的概率为：
$$
\begin{array}{l}
p_{\theta}(\tau) \\
=p\left(s_{1}\right) p_{\theta}\left(a_{1} \mid s_{1}\right) p\left(s_{2} \mid s_{1}, a_{1}\right) p_{\theta}\left(a_{2} \mid s_{2}\right) p\left(s_{3} \mid s_{2}, a_{2}\right) \cdots \\
=p\left(s_{1}\right) \prod_{t=1}^{T} p_{\theta}\left(a_{t} \mid s_{t}\right) p\left(s_{t+1} \mid s_{t}, a_{t}\right)
\end{array}
$$


​		序列τ的回报为每一步的奖励和，称为R(τ)。因此，在Actor的策略为$p(\theta)$的情况下，所能获得的期望奖励为：
$$
\bar{R}_{\theta}=\sum_{\tau} R(\tau) p_{\theta}(\tau)=E_{\tau \sim p_{\theta}(\tau)}[R(\tau)]
$$
​		我们的目的是调整策略$P(\theta)$，使得期望奖励最大化，期望函数就是上面的公式，我们需要采用梯度提升的方法更新参数，所以需要计算参数的梯度，计算过程如下：
$$
\begin{aligned}
\nabla \bar{R}_{\theta}&=\sum_{\tau} R(\tau) \nabla p_{\theta}(\tau)=\sum_{\tau} R(\tau) p_{\theta}(\tau) \frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)}\\
&=\sum_{\tau} R(\tau) p_{\theta}(\tau) \nabla \log p_{\theta}(\tau)\\
&=E_{\tau \sim p_{\theta}(\tau)}\left[R(\tau) \nabla \log p_{\theta}(\tau)\right] \approx \frac{1}{N} \sum_{n=1}^{N} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(\tau^{n}\right)\\
&=\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(a_{t}^{n} \mid s_{t}^{n}\right)
\end{aligned}
$$
​		在第三行中采用N次采样的平均值来近似期望，最后将$p(\theta)$展开。

​		如下图所示为一个完整的PG过程采集数据，然后基于前面得到的梯度提升的式子更新参数，随后再根据更新后的策略再采集数据，再更新参数，如此循环进行。注意到图中的大红字only used once，因为在更新参数后，我们的策略已经变了，而先前的数据是基于更新参数前的策略得到的。

![img](https://pic4.zhimg.com/80/v2-93e3a5fdf36b447c56b76ce2ca4105a2_720w.jpg)	

## 二、Actor-Critic（AC）

​		我们通过一个Critic网络计算Q函数值，用Q值来代替R，则Actor的梯度变为：
$$
\nabla \bar{R}_{\theta}=\frac{1}{N} \sum_{n-1}^{N} \sum_{t=1}^{T_{n}} Q^{\pi_{\theta}}\left(s_{t}^{n}, a_{t}^{n}\right) \nabla \log p_{\theta}\left(a_{t}^{n} \mid s_{t}^{n}\right)
$$
​		Critic是根据估计值和实际值的平方误差进行更新：
$$
\operatorname{loss}=\frac{1}{N} \sum_{n-1}^{N} \sum_{t=1}^{T_{n}}\left(r_{t}^{n}+\max _{a_{t+1}^{n}} Q^{\pi_{\theta}}\left(s_{t+1}^{n}, a_{t+1}^{n}\right)-Q^{\pi_{\theta}}\left(s_{t}^{n}, a_{t}^{n}\right)\right)^{2}
$$


## 三、Advantage Actor-Critic(A2C)

​		给Q值增加一个基线，使得反馈有正有负，这里的基线通常用状态的价值函数来表示，所以梯度变为了：
$$
\nabla \bar{R}_{\theta}=\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}}\left(Q^{\pi_{\theta}}\left(s_{t}^{n}, a_{t}^{n}\right)-V^{\pi_{\theta}}\left(s_{t}^{n}\right)\right) \nabla \log p_{\theta}\left(a_{t}^{n} \mid s_{t}^{n}\right)
$$
​		但是，这样的话我们需要有两个网络分别计算状态-动作价值Q和状态价值V，因此我们做这样的转换：
$$
\begin{array}{l}
Q^{\pi}\left(s_{t}^{n}, a_{t}^{n}\right)=E\left[r_{t}^{n}+V^{\pi}\left(s_{t+1}^{n}\right)\right] \\
Q^{\pi}\left(s_{t}^{n}, a_{t}^{n}\right)=r_{t}^{n}+V^{\pi}\left(s_{t+1}^{n}\right)
\end{array}
$$
​		Critic网络的损失变为实际的状态价值和估计的状态价值的平方损失：
$$
\operatorname{loss}=\frac{1}{N} \sum_{n-1}^{N} \sum_{t=1}^{T_{n}}\left(r_{t}^{n}+V^{\pi}\left(s_{t+1}^{n}\right)-V^{\pi}\left(s_{t}^{n}\right)^{2}\right.
$$

## 三、Asynchronous Advantage Actor-Critic (A3C)



## Reference

[1] [强化学习AC、A2C、A3C算法原理与实现！](https://zhuanlan.zhihu.com/p/51645768)