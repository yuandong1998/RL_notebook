# Lecture 1: Introduction to Reinforcement Learning  
## 1. Introduction



强化学习在不同领域有不同的表现形式：神经科学、心理学、计算机科学、工程领域、数学、经济学等有不同的称呼。

强化学习是机器学习的一个分支：监督学习、无监督学习、强化学习

强化学习的特点：

1. 非监督、只有一个奖励信号。
2. 奖励信号不是实时的，而是延迟的。
3. 时序是很关键的。所以数据不再是独立同分布的数据。
4. 当前agent的action影响后续的数据。

强化学习应用广泛：直升机特技飞行、经典游戏、投资管理、发电站控制、让机器人模仿人类行走等

## 2. 强化学习问题的提出

**奖励 Reward：**

​		reward $R_t$是一个信号标量，表示agent在t步骤时做的如何，agent的任务就是最大化累积的reward。

​		强化学习基于**奖励假设（reward hypothesis）**：所有的目标都可以用期望累积回报的最大化来描述。



**序列决策 Sequential Decision Making ：**

​		目标：选择actions来最大化未来总体的奖励。

​		actions是长期的序列，奖励通常是延迟的，有时为了获得更多的长期回报，最好牺牲眼前的回报。



**个体和环境 Agent & Environment**

在t时刻：

​		agent： 执行$A_t$，接收$O_t$，获得$R_t$。

​		environment：接收$A_t$，发出$O_{t+1}$，发出$R_{t+1}$。



![](https://cdn.mathpix.com/snip/images/r4RoAYgmz3l3-jv-Tn7nSHpSM1yomiF_tWUd40Dd5gI.original.fullsize.png)



**历史和状态 History & State**

​		历史是observations、actions、rewards组成的序列，之后的决策应该由历史决定。

$$
H_t=O_1,R_1,A_1,...,A_{t-1},O_t,R_t
$$
​		状态是决定未来发生什么的信息，是历史的一个函数。历史的数据很大且多数对当前决策无用，所以通过历史得出的状态，改状态拥有决定下一步决策的信息。

$$
S_t=f(H_t)
$$

* 环境状态 Environment State  

  环境状态通常对Agent并不完全可见，也就是Agent有时候并不知道环境状态的所有细节。即使有时候环境状态对Agent可以是完全可见的，这些信息也可能包含着一些无关信息。

* 个体状态 Agent State  

  是Agent的内部呈现，包括Agent可以使用的、决定未来动作的所有信息。Agent状态是强化学习算法可以利用的信息，它可以是历史的一个函数：$S_t^a=f(H_t)$。

* 信息状态 Information State  

  包括历史上所有有用的信息，又称Markov状态。状态$S_t$是马尔可夫的，历史$H_t$也是马尔可夫的。



**完全可观测环境 Fully Observable Environments  **

​		个体能够直接观测到环境状态。在这种条件下：个体对环境的观测 = 个体状态 = 环境状态。正式地说，这种问题是一个马尔科夫决策过程（Markov Decision Process， MDP）



**部分可观测的环境 Partially Observable Environments  **

​		环境是局部可观测的，Agent需要间接观测环境。所以agent state不等于environment state。这种情况被称为：partially observable Markov decision process (POMDP) 。 

​	Agent必须构建自己的state。

* 直接采用历史 ：$S_t^a=H_t$
* Beliefs of environment state，采用已知状态的概率分布作为个体的状态：$S_t^a=(P[S_t^e=s^1],...,P[S_t^e=s^n])$
* RNN，根据前一时刻agent的状态和当前的观测得到当前agent的状态：$S_t^a=\sigma(S_{t-1}^aW_s+O_t W_o)$



## 3. Inside An RL Agent

### 3.1 Agent的组成部分 Major Components of an RL Agent 

​		Agent由一下三个的一个或者多个组成。

**（1）策略 Policy**

​		策略是将state映射为action，分为**确定的策略**（Deterministic policy ）：$a=\pi(s)$和**随机型策略**（Stochastic policy  ）：$\pi(a|s)=P[A_t=1|S_t=S]$



**（2）价值函数 Value Function**

​		价值函数是对未来回报的预测（期望）。表示通过action进入某一状态评价其好坏程度。

$$
v_{\pi}(s)=\mathbb E_\pi[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+...|S_t=s]
$$
**（3）模型 Model**

​		是Agent对环境变化预测的模型，因为Agent希望模拟环境与Agent的交互机制。

* 预测下一个可能状态发生的概率 $P_{ss'}^a=\mathbb P[S_{t+1}=s'|S_t=s,A_t=a]$
* 预测可能获得的即时奖励：$R_s^a=E[R_{t+1}|S_t=s,A_t=a]$



​		模型并不是构建一个个体所必需的，很多强化学习算法中个体并不试图（依赖）构建一个模型。

注：模型仅针对个体而言，环境实际运行机制不称为模型，而称为**环境动力学**(dynamics of environment)，它能够明确确定个体下一个状态和所得的即时奖励。



### 3.2 强化学习个体的分类

可以把个体分为如下三类：

1. 仅基于价值函数的 Value Based：在这样的个体中，有对状态的价值估计函数，但是没有直接的策略函数，策略函数由价值函数间接得到。
2. 仅直接基于策略的 Policy Based：这样的个体中行为直接由策略函数产生，个体并不维护一个对各状态价值的估计函数。
3. 演员-评判家形式 Actor-Critic：个体既有价值函数、也有策略函数。两者相互结合解决问题。



此外，根据个体在解决强化学习问题时是否建立一个对环境动力学的模型，将其分为两大类：

1. 不基于模型的个体: 这类个体并不视图了解环境如何工作，而仅聚焦于价值和/或策略函数。
2. 基于模型的个体：个体尝试建立一个描述环境运作过程的模型，以此来指导价值或策略函数的更新。



### 3.3 学习和规划 Learning & Planning

- 学习：环境初始时是未知的，个体不知道环境如何工作，个体通过与环境进行交互，逐渐改善其行为策略。
- 规划: 环境如何工作对于个体是已知或近似已知的，个体并不与环境发生实际的交互，而是利用其构建的模型进行计算，在此基础上改善其行为策略。



​		一个常用的强化学习问题解决思路是，先学习环境如何工作，也就是了解环境工作的方式，即学习得到一个模型，然后利用这个模型进行规划。



### 3.4 探索和利用 Exploration & Exploitation

​		强化学习类似于一个试错的学习，个体需要从其与环境的交互中发现一个好的策略，同时又不至于在试错的过程中丢失太多的奖励。探索和利用是个体进行决策时需要平衡的两个方面。



### 3.5 预测和控制 Prediction & Control

- 预测：给定一个策略，评价未来。可以看成是求解在给定策略下的价值函数（value function）的过程。How well will I(an agent) do if I(the agent) follow a specific policy?
- 控制：找到一个好的策略来最大化未来的奖励。



## 4. 课程提纲

### 第一部分：强化学习基础理论

1. 强化学习简介： 本讲

2. 马儿可夫决策过程： 理论基础，对于描述强化学习问题很重要

3. 动态规划 小规模强化学习问题的一种解决方案

4. 不基于模型的预测 理论核心

5. 不基于模型的控制 全课重点及核心



### 第二部分：实践中的强化学习

6. 价值函数的近似表示 基于价值函数解决大规模问题的常用技巧

7. 策略梯度方法 基于策略本身解决大规模问题时的常用技巧

8. 整合学习与规划 联合模型解决大规模问题

9. 探索和利用 理论介绍如何平衡探索和利用

10. 案例学习（选） 强化学习在游戏（博弈）中的应用