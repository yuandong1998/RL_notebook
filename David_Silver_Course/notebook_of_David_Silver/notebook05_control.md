# Lecture 5: Model-Free Control 不基于模型的控制  

[TOC]

## 1. Introduction

* 行为策略：指导个体产生与环境进行实际交互行为的策略。

* 目标策略：评价状态或行为价值的策略或者待优化的策略。

* 现时策略学习：个体在学习过程中优化的策略与自己的行为策略是同一个策略。

* 借鉴策略学习：个体在学习过程中优化的策略与自己的行为策略是不同的策略。



## 2. On-Policy Monte-Carlo Control

​		在不基于模型的控制时，我们将无法通过分析、比较基于状态的价值来改善贪婪策略，这是因为基于状态价值的贪婪策略的改善需要知晓状态间转移概率。我们无法事先知道这些状态之间在不同行为下的转移概率，因而无法基于状态价值来改善我们的贪婪策略。  

### 2.1  ϵ-Greedy Exploration ϵ- 贪婪策略


$$
\pi(a | s)=\left\{\begin{array}{ll}
\epsilon / m+1-\epsilon & \text { 如果 } a^{*}=\underset{a \in A}{\operatorname{argmax}} Q(s, a) \\
\epsilon / m & \text { 其它情况 }
\end{array}\right.
$$




### 2.2 Monte-Carlo Control 现时策略蒙特卡罗控制  

​		**现时策略蒙特卡罗控制**：<u>通过 ϵ-贪婪策略采样一个或多个完整的状态序列后，平均得出某一状态行为对的价值，并持续进行策略的评估和改善。通常可以在仅得到一个完整状态序列后就进行一次策略迭代以加速迭代过程。</u>

​	**理论基础**：

​		**GLIE**(greedy in the Limit with Infnite Exploration)  ：它包含两层意思，一是所有的状态行为对会被无限次探索 ；二是另外随着采样趋向无穷多，策略收敛至一个贪婪策略  。存在如下的定理：**GLIE 蒙特卡洛控制能收敛至最优的状态行为价值函数。**如果在使用 ϵ-贪婪策略时，能令 ϵ 随采样次数的无限增加而趋向于 0 就符合 GLIE。  

​	**流程**：

1. 基于给定策略 π，采样第 k 个完整的状态序列 。

2. 对于该状态序列里出现的每一状态行为对 $(S_t,A_t)$，更新其计数 N 和行为价值函数 Q  

   $$
   \begin{array}{c}
   N\left(S_{t}, A_{t}\right) \leftarrow N\left(S_{t}, A_{t}\right)+1 \\
   Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\frac{1}{N\left(S_{t}, A_{t}\right)}\left(G_{t}-Q\left(S_{t}, A_{t}\right)\right)
\end{array}
   $$

 3. 基于新的行为价值函数 Q 以如下方式改善策略

    $$
    \begin{array}{c}
    \epsilon \leftarrow 1 / k \\
    \pi \leftarrow \epsilon-g \operatorname{reed} y(Q)
\end{array}
    $$
    
    
    

## 3. On-Policy Temporal-Difference Learning 现时策略时序差分控制  

### 3.1 Sarsa 算法  

​		**Sarsa算法**：<u>针对一个状态 S，个体通过行为策略产生一个行为 A，执行该行为进而产生一个状态行为对 (S,A)，环境收到个体的行为后会告诉个体即时奖励R 以及后续进入的状态 S’；个体在状态 S’ 时遵循当前的行为策略产生一个新行为 A’，个体此时并不执行该行为，而是通过行为价值函数得到后一个状态行为对 (S’,A’) 的价值，利用这个新的价值和即时奖励 R 来更新前一个状态行为对 (S,A) 的价值。</u>  

​		**迭代公式**：

$$
Q(S, A) \leftarrow Q(S, A)+\alpha\left(R+\gamma Q\left(S^{\prime}, A^{\prime}\right)-Q(S, A)\right)
$$
​		<u>Sarsa算法流程</u>：参数 α 是学习速率参数， γ 是衰减因子。  

![](https://cdn.mathpix.com/snip/images/AaX0bwTi_Dkiw2nOl-75XSfR_wpnTDNldwjBoguHKfc.original.fullsize.png)

​		当行为策略满足前文所述的GLIE 特性同时学习速率参数 α 满足如下时，Sarsa 算法将收敛至最优策略和最优价值函数。  

$$
\sum_{t=1}^{\infty} \alpha_{t}=\infty, \text { 且 } \sum_{t=1}^{\infty} \alpha_{t}^{2}<\infty
$$


### 3.2 Sarsa(λ) 算法  

​		定义 **n-步 Q 收获 (Q-return)** 为：

$$
q_{t}^{(n)}=R_{t+1}+\gamma R_{t+2}+\vdots+\gamma^{n-1} R_{t+n}+\gamma^{n} Q\left(S_{t+n}, A_{t+n}\right)
$$
​		**更新公式**如下：

$$
Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left(q_{t}^{(n)}-Q\left(S_{t}, A_{t}\right)\right)
$$
​		类似于 TD(λ), 可以给 n-步 Q 收获中的每一步收获分配一个权重，并按权重对每一步 Q 收获求和：

$$
q_{t}^{\lambda}=(1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} q_{t}^{(n)}
$$
​		则：

$$
Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left(q_{t}^{(\lambda)}-Q\left(S_{t}, A_{t}\right)\right)
$$
​		反向认识：

$$
\begin{array}{c}
E_{0}(s, a)=0 \\
E_{t}(s, a)=\gamma \lambda E_{t-1}(s, a)+1\left(S_{t}=s, A_{t}=a\right), \gamma, \lambda \in[0,1]
\end{array}
$$

$$
\begin{array}{c}
\left.\delta_{t}=R_{t+1}+\gamma Q\left(S_{t+1}, A_{t+1}\right)-Q\left(S_{t}, A_{t}\right)\right) \\
Q(s, a) \leftarrow Q(s, a)+\alpha \delta_{t} E_{t}(s, a)
\end{array}
$$



​		**$Sarsa(\lambda)$算法流程**：![](https://cdn.mathpix.com/snip/images/uDQmE7QZKegTDuVdF-rTTxggQp_AJdQLdvEA8Kh0fpg.original.fullsize.png)



## 4. Off-Policy Learning

### 4.1 借鉴策略 Q 学习算法  

​		**借鉴策略学习** (off-policy learning) 中产生指导自身行为的策略$\mu(a|s)$与评价策略$\pi(a|s)$是不同的策略。具体地说，个体通过策略 $\mu(a|s)$生成行为与环境发生实际交互，但是在更新这个状态行为对的价值时使用的是目标策略 $\pi(a|s)$。目标策略$\pi(a|s)$多数是已经具备一定能力的策略，例如人类已有的经验或其他个体学习到的经验。借鉴策略学习相当于站在目标策略 $\pi(a|s)$的“肩膀”上学习。  

​		基于蒙特卡洛的借鉴策略学习目前认为仅有理论上的研究价值，在实际中用处不大。这里主要讲解常用借鉴策略 TD 学习。  

$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(\frac{\pi\left(A_{t} | S_{t}\right)}{\mu\left(A_{t} | S_{t}\right)}\left(R_{t+1}+\gamma V\left(S_{t+1}\right)\right)-V\left(S_{t}\right)\right)
$$
​		对于上式，我们可以这样理解：个体处在状态 $S_t$ 中，基于行为策略 µ 产生了一个行为 $A_t$，执行该行为后进入新的状态 $S_{t+1}$，借鉴策略学习要做的事情就是，比较借鉴策略和行为策略在状态 $S_t$ 下产生同样的行为 $A_t$ 的概率的比值，如果这个比值接近 1，说明两个策略在状态  $S_t$ 下采取的行为 $A_t$ 的概率差不多，此次对于状态  $S_t$  价值的更新同时得到两个策略的支持。如果这一概率比值很小，则表明借鉴策略 π 在状态 $S_t$ 下选择 $A_t$ 的机会要小一些，此时为了从借鉴策略学习，我们认为这一步状态价值的更新不是很符合借鉴策略，因而在更新时打些折扣。类似的，如果这个概率比值大于 1，说明按照借鉴策略，选择行为 $A_t$ 的几率要大于当前行为策略产生  $A_t$ 的概率，此时应该对该状态的价值更新就可以大胆些。  



​		借鉴策略 TD 学习中一个典型的行为策略 µ 是基于行为价值函数 Q(s,a)，ϵ-贪婪策略，借鉴策略 π 则是基于 Q(s，a) 的完全贪婪策略，这种学习方法称为 **Q 学习 (Q learning)**。  

​		Q 学习具体的行为**价值更新公式**：  

$$
Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left(R+\gamma \max _{a^{\prime}} Q\left(S_{t+1}, a^{\prime}\right)-Q\left(S_{t}, A_{t}\right)\right)
$$
​		Q 学习的算法流程 ：

![](https://cdn.mathpix.com/snip/images/nwg4xl66zibLgArXECOcfFoZ_a_7GiaTHGZPN0oevrQ.original.fullsize.png)



