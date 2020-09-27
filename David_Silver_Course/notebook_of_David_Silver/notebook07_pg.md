# Lecture 7: Policy Gradient 基于策略梯度的深度强化学习

[TOC]

## 1. Introduction

​		在这一章节我们学习直接参数化策略（policy），如下公式。
$$
\pi_{\theta}(s, a)=\mathbb{P}[a \mid s, \theta]
$$


### 1.1 Value-Based 、Policy-Based 、Actor-Critic的关系：

![](https://cdn.mathpix.com/snip/images/0FfJ8juRIklj-NoCIdzG8mJXY8EAv-Ma9MPFVogtlqs.original.fullsize.png)



### 1.2 Value-Based 、Policy-Based的优缺点

**Policy-Based Advantages：**

* 具有更好的收敛性能
* 对高维或连续的行为空间也有效。（单纯基于价值函数近似的方法无法解决**连续行为空间**的问题）
* 可以学习随机策略（剪刀石头布）



**Policy-Based Disadvantages:**

* 通常收敛到局部最优，而不是全局最优
* 评估一个政策通常是低效的和高方差的



​		基于价值的学习对应的最优策略通常是确定性策略，因为其是从众多行为价值中选择一个最大价值的行为，而有些问题的最优策略却是随机策略，这种情况下同样是无法通过基于价值的学习来求解的。可以看出，基于价值的强化学习虽然能出色地解决很多问题，但面对行为**空间连续、观测受限、随机策略**的学习等问题时仍然显得力不从心。此时基于策略的学习是解决这类问题的一个新的途径。

  

### 1.3 目标函数

1、初始状态价值（start value），$G_1$表示第一步s1的收获。
$$
J_{1}(\theta)=V_{\pi_{\theta}}\left(s_{1}\right)=\mathbb{E}_{\pi_{\theta}}\left[G_{1}\right]
$$

2、平均价值（average value）
$$
J_{\text {av } V}(\theta)=\sum_{s} d^{\pi_{\theta}}(s) V^{\pi_{\theta}}(s)
$$

3、每一时间步的平均奖励（average reward per time-step）
$$
J_{a v R}(\theta)=\sum_{s} d^{\pi_{\theta}}(s) \sum_{a} \pi_{\theta}(s, a) \mathcal{R}_{s}^{a}
$$


​		$d^{\pi_{\theta}}(s)$是基于策略$\pi_\theta$生成的马尔科夫链关于状态的静态分布，并不是指起始状态的分布，而是整体看来基于该策略状态出现概率的分布。对于目标函数的优化有不基于梯度和基于梯度的优化方法，本章关注于基于梯度的优化方法，并且会探索序列的结构。



## 2. Finite Difference Policy Gradient 有限差分策略梯度

### 2.1 梯度下降求解

​	采用梯度下降方法求解，则：

$$
\nabla_{\theta} J(\theta)=\left(\begin{array}{c}
\frac{\partial J(\theta)}{\partial \theta_{1}} \\
\vdots \\
\frac{\partial J(\theta)}{\partial \theta_{n}}
\end{array}\right)
$$

$$
\Delta \theta=\alpha \nabla_{\theta} J(\theta)
$$


​		其中$\nabla_{\theta} J(\theta)$为策略梯度，$\alpha$是步长。我们需要评估$\pi_\theta(s,a)$的策略梯度。对于每一个维度$k\in[1,n]$，评估方式如下，这种方法简单，有噪音，但有时有效。
$$
\frac{\partial J(\theta)}{\partial \theta_{k}} \approx \frac{J\left(\theta+\epsilon u_{k}\right)-J(\theta)}{\epsilon}
$$


### 2.2 Score Function

​		现在我们来计算策略梯度，首先我们假设$\pi_\theta$当非零时是可导的，并且我们知道$\nabla_{\theta} \pi_{\theta}(s, a) $。

$$
\begin{aligned}
\nabla_{\theta} \pi_{\theta}(s, a) &=\pi_{\theta}(s, a) \frac{\nabla_{\theta} \pi_{\theta}(s, a)}{\pi_{\theta}(s, a)} \\
&=\pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a)
\end{aligned}
$$

​		其中$\nabla_{\theta} \log \pi_{\theta}(s, a)$为**score function**。



#### 2.2.1 Softmax 策略

​		Softmax 策略是应用于**离散行为空间**的一种常用策略。该策略使用描述状态和行为的特征$\phi(s, a)$与参数$\theta$的线性组合来权衡一个行为发生的几率：

$$
\pi_{\theta} \propto e^{\phi(s, a)^{T} \theta}
$$
​		对应的分值函数为：

$$
\nabla_{\theta} \log \pi_{\theta}(s, a)=\phi(s, a)-\mathbb{E}_{\pi_{\theta}}[\phi(s, \cdot)]
$$




#### 2.2.2 高斯策略

​		高斯策略是应用于**连续行为空间**的一种常用策略。该策略对应的行为从高斯分布$\mathbb{N}\left(\mu(s), \sigma^{2}\right)$中产生。其均值 $\mu(s)=\phi(s)^{T} \theta$。高斯策略对应的分值函数为:

$$
\nabla_{\theta} \log \pi_{\theta}(s, a)=\frac{(a-\mu(s)) \phi(s)}{\sigma^{2}}
$$


###　2.3 One-Step MDPs

​		假设现在有一个单步马尔科夫决策过程，对应的强化学习问题是个体与环境每产生一个行为交互一次即得到一个即时奖励$r=R_{s,a}$，并形成一个完整的状态序列。目标函数为：
$$
\begin{aligned}
J(\theta) &=\mathbb{E}_{\pi_{\theta}}[r] \\
&=\sum_{s \in S} d(s) \sum_{a \in A} \pi_{\theta}(s, a) R_{s, a}
\end{aligned}
$$
​		

​		目标函数的梯度为：
$$
\begin{aligned}
\nabla_{\theta} J(\theta) &=\sum_{s \in S} d(s) \sum_{a \in A} \nabla_{\theta} \pi_{\theta}(s, a) R_{s, a} \\
&=\sum_{s \in S} d(s) \sum_{a \in A} \pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a) R_{s, a} \\
&=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) r\right]
\end{aligned}
$$



### 2.4 Policy Gradient Theorem  策略梯度定理		

​		 策略梯度定理扩展到多步的MDP，将即时奖励r替换为长期价值$Q^\pi(s,a)$。

​		存在如下的策略梯度定理：对于任何可微的策略函数$\pi_\theta(s,a)$ 以及三种策略目标函数$J_1,J_{avV},\frac{1}{1-\gamma}J_{avR}$中的任意一种来说，策略目标函数的梯度 (策略梯度) 都可以写成用分值函数表示的形式：
$$
\nabla_{\theta} J(\theta)=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) Q_{\pi_{\theta}}(s, a)\right]
$$



## 3. Monte-Carlo Policy Gradient 蒙特卡洛策略梯度

​		该算法实际应用不多，主要是由于其需要完整的状态序列来计算收获，同时用收获来代替行为价值也存在较高的变异性，导致许多次的参数更新的方向有可能不是真正策略梯度的方向。其中的$v_t$为累积回报，即为之前的$G_t$。

<img src="https://cdn.mathpix.com/snip/images/aqC54CyoGbckr5RZ1X7853uRQ7ZMtt00g6x_OanWVTM.original.fullsize.png" style="zoom:67%;" />



## 4. Actor-Critic Policy Gradient

### 	4.1 QAC算法

 		蒙特卡洛算法有很高的方差，所以提出了Actor-Critic算法。Actor-Critic 算法包含一个策略函数和行为价值函数。其中策略函数充当演员 (Actor), 生成行为与环境交互；行为价值函数充当(Critic)，负责评价演员的表现，并指导演员的后续行为动作。Critic 的行为价值函数是基于策略$\pi_\theta$的一个近似：
$$
Q_{w}(s, a) \approx Q_{\pi_{\theta}}(s, a)
$$
​		critic是要解决策略评估问题，这个问题在前几周讨论过，有（1）Monte-Carlo，（2）Temporal-Difference，（3）TD（$\lambda$） 

​		假设：用线性价值函数来近似，critic根据TD(0)更新参数，Actor根据策略梯度更新参数，如下为QAC算法：

![](https://cdn.mathpix.com/snip/images/-6gfKOsHU-Ys3Ksea1vz5wqLcGcCWY-RbaGCmEmxu_M.original.fullsize.png)



### 4.2 Compatible Function Approximation  相容函数逼近定理

​		简单的 QAC 算法虽然不需要完整的状态序列，但是由于引入的 Critic 仍然是一个近似价值函数，存在着引入偏差的可能性，不过当价值函数接受的输入的特征和函数近似方式足够幸运时，可以避免这种偏差而完全遵循策略梯度的方向。

​	**定理**：如果下面两个条件满足：

1. 近似价值函数的梯度与分值函数的梯度相同，即：
   $$
   \nabla_{w} Q_{w}(s, a)=\nabla_{\theta} \log \pi_{\theta}(s, a)
   $$
   
2. 近似价值函数的参数 w 能够最小化
   $$
   \epsilon=\mathbb{E}_{\pi_{\theta}}\left[\left(Q_{\pi_{\theta}}(s, a)-Q_{w}(s, a)\right)^{2}\right]
   $$



​	如果满足上面的两个条件，那么策略梯度 ∇ θ J(θ) 是准确的，即:
$$
\nabla_{\theta} J(\theta)=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) Q_{w}(s, a)\right]
$$


**证明：**

![](https://cdn.mathpix.com/snip/images/E5XRR7ui4761T_J35jRnXcQZgirxTdUn29xB5EAky5w.original.fullsize.png)

### 4.3 Reducing Variance Using a Baseline  

​		我们从策略梯度中减去一个基线函数B(s)，可以在不改变期望的情况下减少方差，因为如下的公式为0 。
$$
\begin{aligned}
\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) B(s)\right] &=\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a} \nabla_{\theta} \pi_{\theta}(s, a) B(s) \\
&=\sum_{s \in \mathcal{S}} d^{\pi_{\theta}} B(s) \nabla_{\theta} \sum_{a \in \mathcal{A}} \pi_{\theta}(s, a) \\
&=0
\end{aligned}
$$
​	

​		一个好的基线是状态值函数$B(s)=V^{\pi_{\theta}}(s)$，则策略梯度改写为如下的，使得方差减小。
$$
\begin{aligned}
A^{\pi \theta}(s, a) &=Q^{\pi_{\theta}}(s, a)-V^{\pi_{\theta}(s)} \\
\nabla_{\theta} J(\theta) &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) A^{\pi_{\theta}}(s, a)\right]
\end{aligned}
$$


​		对于$V^{\pi_\theta}(s)$的TD error，是对优势函数的无偏估计。
$$
\delta^{\pi_{\theta}}=r+\gamma V^{\pi_{\theta}}\left(s^{\prime}\right)-V^{\pi_{\theta}}(s)
$$

$$
\begin{aligned}
\mathbb{E}_{\pi_{\theta}}\left[\delta^{\pi_{\theta}} \mid s, a\right] &=\mathbb{E}_{\pi_{\theta}}\left[r+\gamma V^{\pi_{\theta}}\left(s^{\prime}\right) \mid s, a\right]-V^{\pi_{\theta}}(s) \\
&=Q^{\pi_{\theta}}(s, a)-V^{\pi_{\theta}}(s) \\
&=A^{\pi_{\theta}}(s, a)
\end{aligned}
$$

​		所以可以计算如下策略梯度：
$$
\nabla_{\theta} J(\theta)=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \delta^{\pi_{\theta}}\right]
$$
​		可以用参数来估计TD error，只需要一组参数v 。
$$
\delta_{v}=r+\gamma V_{v}\left(s^{\prime}\right)-V_{v}(s)
$$

### 4.4 Policy Gradient with Eligibility Traces  

​		可以像后向$TD(\lambda)$一样引入eligibility traces  。
$$
\begin{aligned}
\delta &=r_{t+1}+\gamma V_{v}\left(s_{t+1}\right)-V_{v}\left(s_{t}\right) \\
e_{t+1} &=\lambda e_{t}+\nabla_{\theta} \log \pi_{\theta}(s, a) \\
\Delta \theta &=\alpha \delta e_{t}
\end{aligned}
$$

### 4.5 Natural Policy Gradient  

$$
\nabla_{\theta}^{n a t} \pi_{\theta}(s, a)=G_{\theta}^{-1} \nabla_{\theta} \pi_{\theta}(s, a)
$$

​		Gθ is the Fisher information matrix ：
$$
G_{\theta}=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a)^{T}\right]
$$


### 4.6  Natural Actor-Critic  

​		Using compatible function approximation ，
$$
\nabla_{w} A_{w}(s, a)=\nabla_{\theta} \log \pi_{\theta}(s, a)
$$
​		So the natural policy gradient simplifies  ,
$$
\begin{aligned}
\nabla_{\theta} J(\theta) &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) A^{\pi_{\theta}}(s, a)\right] \\
&=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a)^{T} w\right] \\
&=G_{\theta} w \\
\nabla_{\theta}^{n a t} f(\theta) &=w
\end{aligned}
$$


## 5. 深度确定性策略梯度 (DDPG) 算法

​		深度确定性策略梯度算法是使用深度学习技术、同时基于 Actor-Critic 算法的确定性策略算法。该算法中的 Actor 和 Critic 都使用深度神经网络来建立近似函数。由于该算法可以直接从Actor 的策略生成确定的行为而不需要依据行为的概率分布进行采样而被称为确定性策略。该算法在学习阶段通过在确定性的行为基础上增加一个噪声函数而实现在确定性行为周围的小范围内探索。此外，该算法还为 Actor 和 Critic 网络各备份了一套参数用来计算行为价值的期待值以更稳定地提升 Critic 的策略指导水平。使用备份参数的网络称为目标网络，其对应的参数每次更新的幅度很小。另一套参数对应的 Actor 和 Critic 则用来生成实际交互的行为以及计算相应的策略梯度，这一套参数每学习一次就更新一次。这种双参数设置的目的是为了减少因近似数据的引导而发生不收敛的情形。这四个网络具体使用的情景为：  

1. Actor 网络：根据当前状态 s0 生成的探索或不探索的具体行为 a0；
2. Target Actor 网络：根据环境给出的后续状态 s1 生成预估价值用到的 a1；
3. Critic 网络：计算状态 s0 和生成的行为 a0 对应的行为价值；
4. Target Critic 网络：根据后续状态 s1,a1 生成用来计算目标价值 y = Q(s0; a0) 的 Q′(s1; a1)；



​	DDPG 算法表现出色，能较为稳定地解决连续行为空间下强化学习问题，其具体流程如下所示。  

![](https://cdn.mathpix.com/snip/images/WvvGSnOpYIeCrTKXV2x-Y_NyA8Zgz13Mf9l1uw2wvUA.original.fullsize.png)

