# Lecture 2: Markov Decision Processes  

[TOC]

## 1.Markov Process 马尔科夫过程

### 1.1 Introduction to MDPs  

​		**马尔科夫决策过程**：可以对**完全可观测环境**进行描述，几乎所有的强化学习问题都可以转为MDP。



### 1.2 Markov Property 马尔科夫特性  

​		马尔科夫特性：The future is independent of the past given the present 。定义：$\mathbb P[S_{t+1}|S_t]=P[S_{t+1}|S_1,...,S_t]$



### 1.3 State Transition Matrix  状态转移矩阵

$$
P_{ss'}=\mathbb P[S_{t+1}=s'|S_t=s]
$$



### 1.4 Markov Process 马尔科夫过程

​		**定义**：马尔可夫过程是一个无记忆的随机过程，即一个随机状态序列具有马尔可夫性。可以用一个元组<S,P>表示，其中S是有限数量的状态集，P是状态转移概率矩阵。



​		**Sample Episodes：**从起始态C1到结束态Sleep的所有可能的过程。



![](https://cdn.mathpix.com/snip/images/Mm0cozScUaYy9QVQI6P_bku4Oenl6SgQ0_bSjbOM6EQ.original.fullsize.png)





## 2.Markov Reward Process 马尔科夫奖励过程

### 2.1 Define 定义

​		A Markov reward process is a Markov chain with values 。定义：马尔科夫奖励过程可以表示为元组$<S,P,R,\gamma>$

* S是一个有限状态集合

* P是状态转移矩阵

* R是一个奖励函数（reward function），表示在t时状态为s的奖励$R_{t+1}$。$\mathbb R_s=E[R_{t+1}|S_t=s]$

  > 很多听众纠结为什么奖励是t+1时刻的。照此理解起来相当于离开这个状态才能获得奖励而不是进入这个状态即获得奖励。David指出这仅是一个约定，为了在描述RL问题中涉及到的观测O、行为A、和奖励R时比较方便。他同时指出如果把奖励改为 $R_t$而不是 $R_{t+1}$，只要规定好，本质上意义是相同的，在表述上可以把奖励描述为“当进入某个状态会获得相应的奖励”。

* $\gamma$是一个衰减系数（Discount Factor），用来反应更注重现在的reward还是未来的reward。



### 2.2 Return 收获

​		**收获$G_t$**：<u>为在一个马尔科夫奖励链上从t时刻开始往后所有的奖励的有衰减的总和，收获是相对于一个样本来说的。</u>

$$
G_t=R_{t+1}+\gamma R_{t+2}+...=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}
$$


问：为什么需要衰减因子？

* 在数学上是方便的
* 防止无限的reward（循环）
* 在金融上即时奖励可能比延迟奖励获得更多的利息
* 动物/人类的行为表现出对即时回报的偏好



### 2.3 Value Function 价值函数

​		**价值函数**：<u>在马尔科夫决策过程下基于策略 π 的状态价值函数，表示从状态 s开始，遵循当前策略$\pi$ 时所获得的收获的期望。价值函数是对于一个策略而言的。</u>

$$
v(s)=\mathbb E[G_t|S_t=s]
$$


### 2.4 Bellman Equation for MRPs   贝尔曼方程

​		如下公式，价值函数可以被分解为两个部分：

$$
\begin{aligned}
v(s) &=\mathbb{E}\left[G_{t} | S_{t}=s\right] \\
&=\mathbb{E}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\ldots | S_{t}=s\right] \\
&=\mathbb{E}\left[R_{t+1}+\gamma\left(R_{t+2}+\gamma R_{t+3}+\ldots\right) | S_{t}=s\right] \\
&=\mathbb{E}\left[R_{t+1}+\gamma G_{t+1} | S_{t}=s\right] \\
&=\mathbb{E}\left[R_{t+1}+\gamma v\left(S_{t+1}\right) | S_{t}=s\right]
\end{aligned}
$$


​		一个是该状态的即时奖励期望，也就是即时奖励；另一个是下一时刻状态的价值期望，可以根据下一时刻状态的概率分布得到其期望。如果用$s'$表示s状态下一时刻任一可能状态，那么贝尔曼方程可写为：

$$
v(s)=R_s+\gamma \sum_{s'\in S}P_{s s'}v(s')
$$




​		用矩阵可以表示为：$v=R+\gamma P v$，其中：

$$
\left[\begin{array}{c}
v(1) \\
\vdots \\
v(n)
\end{array}\right]=\left[\begin{array}{c}
\mathcal{R}_{1} \\
\vdots \\
\mathcal{R}_{n}
\end{array}\right]+\gamma\left[\begin{array}{ccc}
\mathcal{P}_{11} & \dots & \mathcal{P}_{1 n} \\
\vdots & & \\
\mathcal{P}_{n1} & \dots & \mathcal{P}_{n n}
\end{array}\right]\left[\begin{array}{c}
v(1) \\
\vdots \\
v(n)
\end{array}\right]
$$


 		求解贝尔曼方程：

$$
\begin{aligned}
v &=\mathcal{R}+\gamma \mathcal{P} v \\
(I-\gamma \mathcal{P}) v &=\mathcal{R} \\
v &=(I-\gamma \mathcal{P})^{-1} \mathcal{R}
\end{aligned}
$$


对于n个states，该方式复杂度为$O(n^3)$，只可以对小MRPs方法，对于大规模MRPs问题可以采用：

* Dynamic programming 动态规划
* Monte-Carlo evaluation 蒙特卡洛评价
* Temporal-Difference learning  



## 3.Markov Decision Process 马尔科夫决策过程

### 3.1 MDP 定义

​		A Markov decision process (MDP) is a Markov reward process with decisions. It is an environment in which all states are Markov，定义：马尔科夫决策过程可以表示为元组$<S,A,P,R,\gamma>$

* S是一个有限状态集合
* A是一个有限决策集合
* P是状态转移矩阵，$P_{s s'}^a=P[S_{t+1}=s'|S_t=s,A_t=a]$
* R是一个奖励函数（reward function），表示在t时状态为s的奖励$R_{t+1}$。$\mathbb R_s^a=E[R_{t+1}|S_t=s,A_t=a]$
* $\gamma$是一个衰减系数（Discount Factor），用来反应更注重现在的reward还是未来的reward。



### 3.2 Policies 策略

​		策略$\pi$：<u>是给定一个states，actions的分布。只依赖于当前状态，不依赖于历史。现在只考虑静态的policies，与时间无关。</u>

$$
\pi(a|s)=\mathbb P[A_t=a|S_t=s]
$$


​		给定一个MDP $M=<S,A,P,R,\gamma>$和策略$\pi$，则状态序列是一个马尔科夫过程$<S,P^\pi>$，状态和奖励序列是一个马尔科夫奖励过程$<S,P^\pi,R^\pi,\gamma>$ 其中：

$$
\begin{aligned}
\mathcal{P}_{s, s^{\prime}}^{\pi} &=\sum_{a \in \mathcal{A}} \pi(a | s) \mathcal{P}_{s s^{\prime}}^{a} \\
\mathcal{R}_{s}^{\pi} &=\sum_{a \in \mathcal{A}} \pi(a | s) \mathcal{R}_{s}^{a}
\end{aligned}
$$




### 3.3 Value Function 价值函数

​		**state-value function 状态价值函数**$v_\pi(s)$：<u>表示从状态s开始，**遵循当前策略**时所获得的收获的期望；或者说在执行当前策略π时，衡量个体处在状态s时的价值大小。</u>

$$
v_\pi(s)=\mathbb E_\pi[G_t|S_t=s]
$$


**action-value function   行为价值函数**$q_\pi(s,a)$：<u>**表示**在执行策略π时，对当前状态s执行某一具体行为a所能的到的收获的期望；或者说在遵循当前策略π时，衡量对当前状态执行行为a的价值大小。</u>
$$
q_\pi(s,a)=\mathbb E_\pi[G_t|S_t=s,A_t=a]
$$


### 3.4 Bellman Expectation Equation  贝尔曼期望方程

​		可以得出一下两个方程：

$$
v_{\pi}(s)=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) | S_{t}=s\right]
$$

$$
q_{\pi}(s, a)=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma q_{\pi}\left(S_{t+1}, A_{t+1}\right) | S_{t}=s, A_{t}=a\right]
$$



​		推导：

$$
v_{\pi}(s)=\sum_{a \in \mathcal{A}} \pi(a | s) q_{\pi}(s, a)
$$

$$
q_{\pi}(s, a)=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{\pi}\left(s^{\prime}\right)
$$

$$
v_{\pi}(s)=\sum_{a \in \mathcal{A}} \pi(a | s)\left(\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{\pi}\left(s^{\prime}\right)\right)
$$

$$
q_{\pi}(s, a)=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} \sum_{a^{\prime} \in \mathcal{A}} \pi\left(a^{\prime} | s^{\prime}\right) q_{\pi}\left(s^{\prime}, a^{\prime}\right)
$$





​		可得矩阵表达式：$v_{\pi}=\mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi} v_{\pi}$，直接解为：$v_{\pi}=\left(I-\gamma \mathcal{P}^{\pi}\right)^{-1} \mathcal{R}^{\pi}$



### 3.5 Optimal Value Function 最优值函数

​		**最优状态值函数$v_∗(s)$**：<u>是所有策略下产生的众多状态价值函数中的最大者：</u>  

$$
v_{*}(s)=\max _{\pi} v_{\pi}(s)
$$


​		 **最优行动价值函数$q_*(s,a)$**：<u>是所有策略下产生的众多行为价值函数中的最大者：</u>  

$$
q_{*}(s,a)=\max _{\pi} q_{\pi}(s,a)
$$




### 3.6 Optimal Policy 最优策略

定义一个偏序：$\pi \geq \pi^{\prime}$ if $v_{\pi}(s) \geq v_{\pi^{\prime}}(s), \forall s$ 。**定理** 对于任何MDP，下面几点成立：

1. 存在一个最优策略，比任何其他策略更好或至少相等；

2. 所有的最优策略有相同的最优价值函数；

3. 所有的最优策略具有相同的行为价值函数。



​		可以通过最大化最优行为价值函数来找到最优策略：

$$
\pi_{*}(a | s)=\left\{\begin{array}{ll}
1 & \text { if } a=\underset{a \in \mathcal{A}}{\operatorname{argmax}} q_{*}(s, a) \\
0 & \text { otherwise }
\end{array}\right.
$$




​		对于任何MDP问题，总存在一个确定性的最优策略；同时如果我们知道最优行为价值函数，则表明我们找到了最优策略。

推理：

$$
v_{*}(s)=\max _{a} q_{*}(s, a)\\
q_{*}(s, a)=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)\\
v_{*}(s)=\max _{a} \mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)\\
q_{*}(s, a)=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} \max _{a^{\prime}} q_{*}\left(s^{\prime}, a^{\prime}\right)
$$
​		Bellman最优方程是非线性的，没有固定的解决方案，通过一些迭代方法来解决：价值迭代、策略迭代、Q学习、Sarsa等。



## 4.Extensions to MDPs

###  4.1 Infinite and continuous MDPs  无限状态或连续MDP

### 4.2 Partially observable MDPs  部分可观测MDP

### 4.3 Undiscounted, average reward MDPs  非衰减、平均奖励MDP

