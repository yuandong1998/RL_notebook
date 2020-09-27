# Lecture 6: Value Function Approximation 价值函数的近似表示  

[TOC]

## 1. Introduction

​		本章解决状态数量多和连续状态的强化学习问题。**通过参数w的函数来近似状态价值函数和状态行为对价值函数，在MC或者TD算法中通过更新w参数来实现**。

​		三种构建近似函数的方法：

![](https://cdn.mathpix.com/snip/images/7GnJItI_y-ybf-jgkX6VAFbitVNrvzKof99ry-EFX2s.original.fullsize.png)

​		需要一个训练方法适合于非独立同分布的数据。



## 2. Incremental Methods

### 2.1 梯度下降

略

### 2.2 线性函数近似

略

### 2.3 目标函数  

​	对于**MC学习**：

$$
\begin{array}{c}
J(w)=\frac{1}{2 M} \sum_{t=1}^{M}\left[G_{t}-\hat{V}\left(S_{t}, w\right)\right]^{2} \\
J(w)=\frac{1}{2 M} \sum_{t=1}^{M}\left[G_{t}-\hat{Q}\left(S_{t}, A_{t}, w\right)\right]^{2}
\end{array}
$$
​	对于 **TD(0)** 和**反向认识 TD(λ)** 学习  ：

$$
\begin{array}{c}
J(w)=\frac{1}{2 M} \sum_{t=1}^{M}\left[\left(R_{t}+\gamma \hat{V}\left(S_{t}^{\prime}, w\right)\right)-\hat{V}\left(S_{t}, w\right)\right]^{2} \\
J(w)=\frac{1}{2 M} \sum_{t=1}^{M}\left[\left(R_{t}+\gamma \hat{Q}\left(S_{t}^{\prime}, A_{t}^{\prime}, w\right)\right)-\hat{Q}\left(S_{t}, A_{t}, w\right)\right]^{2}
\end{array}
$$
​	**前向认识 TD(λ)** 学习  ：

$$
\begin{array}{c}
J(w)=\frac{1}{2 M} \sum_{t=1}^{M}\left[G_{t}^{\lambda}-\hat{V}\left(S_{t}, w\right)\right]^{2} \\
J(w)=\frac{1}{2 M} \sum_{t=1}^{M}\left[q_{t}^{\lambda}-\hat{Q}\left(S_{t}, A_{t}, w\right)\right]^{2}
\end{array}
$$

## 3. Batch Methods

### 3.1 DQN算法

​		DQN 算法主要使用**经历回放 (experience replay) 来实现价值函数的收敛**。其具体做法为：个体能记住既往的状态转换经历，对于每一个完整状态序列里的每一次状态转换，依据当前状态的$s_t$价值以 ϵ-贪婪策略选择一个行为$a_t$，执行该行为得到奖励$r_{t+1}$和下一个状态$s_{t+1}$ ，将得到的状态转换存储至记忆中，当记忆中存储的容量足够大时，随机从记忆力提取一定数量的状态转换，用状态转换中下一状态来计算当前状态的目标价值，使用公式 (6.4) 计算目标价值与网络输出价值之间的均方差代价，使用小块梯度下降算法更新网络的参数。

![](https://cdn.mathpix.com/snip/images/mCGcoEKCBkRRu_LY4glh8pDQC0NolS7FalfwpiZFsho.original.fullsize.png)



### 3.2 DDQN算法

​		DQN 算法在深度强化学习领域取得了不俗的成绩，不过其并不能保证一直收敛，研究表明这种估计目标价值的算法过于乐观的高估了一些情况下的行为价值，导致算法会将次优行为价值一致认为最优行为价值，最终不能收敛至最佳价值函数。一种使用双价值网络的 DDQN(double deep Q network) 被认为较好地解决了这个问题。该算法使用两个架构相同的近似价值函数，其中一个用来根据策略生成交互行为并随时频繁参数 ($\theta$)，另一个则用来生成目标价值, 其参数 ($\theta^-$) 每隔一定的周期进行更新。该算法绝大多数流程与 DQN 算法一样，只是在更新目标价值时使用公式：	

$$
Q_{\text {target}}\left(S_{t}, A_{t}\right)=R_{t}+\gamma Q\left(S_{t}^{\prime}, \max _{a^{\prime}} Q\left(S_{t}^{\prime}, a^{\prime} ; \theta\right) ; \theta^{-}\right)
$$
​		该式表明，DDQN 在生成目标价值时使用了生成交互行为并频繁更新参数的价值网络$Q(\theta)$，在这个价值网络中挑选状态$S'$下最大价值对应的行为$A'_t$ ，随后再用状态行为对$(S'_t,A'_t)$代入目标价值网络$Q(\theta^-)$得出目标价值。

