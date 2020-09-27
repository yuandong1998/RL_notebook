#  Lecture04：Model-Free Prediction 不基于模型的预测  

[TOC]

## 1. Introduction 简介

​		如何解决一个可以被认为是 MDP、**但却不掌握MDP 具体细节**的问题，也就是讲述个体如何在没有对环境动力学认识的模型的条件下如何直接通过个体与环境的实际交互来评估一个策略的好坏或者寻找到最优价值函数和最优策略。  本章分为三个部分，将分别从理论上阐述：

* **基于完整采样的蒙特卡罗强化学习**
* **基于不完整采样的时序差分强化学习**
* **介于两者之间的 λ 时序差分强化学习**。  



## 2. Monte-Carlo Learning 蒙特卡罗强化学习  

​		**蒙特卡罗强化学习** (Monte-Carlo reinforcement learning, MC 学习)： <u>指在不清楚 MDP 状态转移概率的情况下，直接从经历完整的状态序列 (episode) 来估计状态的真实价值，并认为某状态的价值等于在多个状态序列中以该状态算得到的所有收获的平均</u>。  



​	**累进更新平均值（incremental mean)**的计算如下所示：
$$
\begin{aligned}
\mu_{k} &=\frac{1}{k} \sum_{j=1}^{k} x_{j} \\
&=\frac{1}{k}\left(x_{k}+\sum_{j=1}^{k-1} x_{j}\right) \\
&=\frac{1}{k}\left(x_{k}+(k-1) \mu_{k-1}\right) \\
&=\mu_{k-1}+\frac{1}{k}\left(x_{k}-\mu_{k-1}\right)
\end{aligned}
$$


​	**递增式的蒙特卡罗法更新状态价值公式**：

$$
\begin{array}{c}
N\left(S_{t}\right) \leftarrow N\left(S_{t}\right)+1 \\
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\frac{1}{N\left(S_{t}\right)}\left(G_{t}-V\left(S_{t}\right)\right)
\end{array}
$$


​		在一些实时或者无法统计准确状态被访问次数时，可以用一个系数 α 来代替状态计数的倒数，此时公式变为：  

$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}-V\left(S_{t}\right)\right)
$$




## 3. Temporal-Difference Learning 时序差分强化学习  

​		**时序差分强化学习** (temporal-difference reinforcement learning, TD 学习)：<u>指从采样得到的不完整的状态序列学习，该方法通过合理的引导（bootstrapping），先估计某状态在该状态序列完整后可能得到的收获，并在此基础上利用前文所属的累进更新平均值的方法得到该状态的价值，再通过不断的采样来持续更新这个价值。</u>  

​		具体地说，在 TD 学习中，算法在估计某一个状态的收获时，用的是离开该状态的即刻奖励$R_{t+1}$ 与下一时刻状态 $S_{t+1}$ 的预估状态价值乘以衰减系数$\gamma$组成：  

$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)\right)
$$


​		其中$R_{t+1}+\gamma V\left(S_{t+1}\right)$称为 **TD 目标值** ，$R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)$称为**TD 误差**。

​		引导 (bootstrapping)：指的是用 TD 目标值代替收获$G_t$ 的过程。



​		**MC与TD算法的区别**

​		TD 学习在更新状态价值时使用的是 TD 目标值，即基于即时奖励和下一状态的预估价值来替代当前状态在状态序列结束时可能得到的收获，它是当前状态价值的有偏估计，而 MC 学习则使用实际的收获来更新状态价值，是某一策略下状态价值的无偏估计。  

​		TD 算法使用了MDP 问题的马儿可夫属性，在具有马尔科夫性的环境下更有效；但是 MC 算法并不利用马儿可夫属性，适用范围不限于具有马尔科夫性的环境。

  

>**TD 学习能比 MC 学习更快速灵活的更新状态的价值估计**，这在某些情况下有着非常重要的实际意义。回到驾车返家这个例子中来，我们给驾车返家制定一个新的目标，不再以耗时多少来评估状态价值，而是要求安全平稳的返回家中。假如有一次你在驾车回家的路上突然碰到险情：对面开过来一辆车感觉要和你迎面相撞，严重的话甚至会威胁生命，不过由于最后双方驾驶员都采取了紧急措施没有让险情实际发生，最后平安到家。如果是使用蒙特卡罗学习，路上发生的这一险情可能引发的极大负值奖励将不会被考虑，你不会更新在碰到此类险情时的状态的价值；但是在 TD 学习时，碰到这样的险情过后，你会立即大幅调低这个状态的价值，并在今后再次碰到类似情况时采取其它行为，例如降低速度等来让自身处在一个价值较高的状态中，尽可能避免发生意外事件的发生。  



## 4. TD($\lambda$)

​		**n-步预测** <u>指从状态序列的当前状态 ($S_t$) 开始往序列终止状态方向观察至状态$S_{t+n-1}$，使用这 n 个状态产生的即时奖励 $(R_{t+1}; R_{t+2};...;R_{t+n})$ 以及状态$S_{t+n}$ 的预估价值来计算当前第状态$S_t$ 的价值。</u>

​		定义 **n-步收获**为：  

$$
G_{t}^{(n)}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{n-1} R_{t+n}+\gamma^{n} V\left(S_{t+n}\right)
$$


​		n-步 TD 学习对应的状态价值函数的**更新公式**为：  

$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}^{(n)}-V\left(S_{t}\right)\right)
$$


​		为了能在不增加计算复杂度的情况下综合考虑所有步数的预测，我们引入了一个新的参数λ，并定义λ-收获为从 n=1 到无穷的所有步收获的权重之和。  
$$
G_{t}^{\lambda}=(1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t}^{(n)}
$$
​		对应的 TD($\lambda$) 的更新公式为 :

$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}^{(\lambda)}-V\left(S_{t}\right)\right)
$$


​		随着n的增大，其 n-收获的权重呈几何级数的衰减。当在 T 时刻到达终止状态时，未分配的权重 (右侧阴影部分) 全部给予终止状态的实际收获值。  



​	**前向认识 TD($\lambda$)**

​		什么是前向认识：<u>更新一个状态的价值需要知道所有后续状态的价值，这和 MC 算法的要求一样，当 λ = 1 时对应的就是 MC 算法，这个实际计算带来了不便。</u>  



​	**反向认识 TD($\lambda$)**

​		什么是效用迹：<u>如果把老鼠遭到电击的原因认为是之前接受了较多次数的响铃，则称这种归因为频率启发(frequency heuristic) 式；而把电击归因于最近少数几次状态的影响，则称为就近启发 (recncyheuristic) 式。如果给每一个状态引入一个数值： 效用 (eligibility, E) 来表示该状态对后续状态的影响，就可以同时利用到上述两个启发。而所有状态的效用值总称为效用迹 (eligibility traces,ES)。</u>  

​		效用迹公式：

$$
\begin{array}{c}
E_{0}(s)=0 \\
E_{t}(s)=\gamma \lambda E_{t-1}(s)+1\left(S_{t}=s\right), \gamma, \lambda \in[0,1]
\end{array}
$$
​		针对每一个状态存在一个 E 值，且 E 值并不需要等到状态序列到达终止状态才能计算出来，它是根据已经经过的状态序列来计算得到，并且在每一个时刻都对每一个状态进行一次更新，E 值存在饱和现象，有一个瞬时最高上限：  

$$
E_{s a t}=\frac{1}{1-\gamma \lambda}
$$
​		在更新状态价值时把该状态的效用同时考虑进来，价值更新可以表示为 ：

$$
\begin{aligned}
\delta_{t}=&\left(R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)\right) \\
& V(s) \leftarrow V(s)+\alpha \delta_{t} E_{t}(s)
\end{aligned}
$$


* 当$\lambda =0$时，$S_t=s$一直成立，此时价值更新等同于 TD(0) 算法。

* 当$\lambda =1$时，可以每经历一个状态就更新状态的价值，这种实时更新的方法并不完全等同于
  MC。

* 当$\lambda\in(0,1)$时，前向认识和反向认识完全等效，但在实时学习时存在差别。

