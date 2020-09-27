# Lecture 8: Integrating Learning and Planning  

[TOC]

## 一、Introduction

​		多数强化学习问题可以通过表格式或基于近似函数来直接学习状态价值或策略函数，在这些学习方法中，个体并不试图去理解环境动力学。如果能建立一个较为准确地模拟环境动力学特征的模型或者问题的模型本身就类似于一些棋类游戏是明确或者简单的，个体就可以通过构建这样的模型来模拟其与环境的交互，这种依靠模型模拟而不实际与环境交互的过程类似于“**思考**”过程。通过思考，个体可以对问题进行规划、在与环境实际交互时搜索交互可能产生的各种后果并从中选择对个体有利的结果。**这种思想可以广泛应用于规则简单、状态或结果复杂的强化学习问题中。**

![](https://cdn.mathpix.com/snip/images/iv_y1opApBnKJ6pH9GNUZDte62ot_eriAGD21lcd38g.original.fullsize.png)

​		模型M是一个马尔科夫决策过程MDP$<S,A,P,R>$的参数化形式，来描述状态转换和奖励函数。
$$
\begin{array}{l}
S_{t+1} \sim P_{\eta}\left(S_{t+1} \mid S_{t}, A_{t}\right) \\
R_{t+1}=R_{\eta}\left(R_{t+1} \mid S_{t}, A_{t}\right)
\end{array}
$$
​		通常**假设状态转换和奖励之间条件独立**：
$$
\mathbb{P}\left[S_{t+1}, R_{t+1} \mid S_{t}, A_{t}\right]=\mathbb{P}\left[S_{t+1} \mid S_{t}, A_{t}\right] \mathbb{P}\left[R_{t+1} \mid S_{t}, A_{t}\right]
$$


## 二、Model-Based Reinforcement Learning  

### 2.1 优缺点

优点：

* 可以通过监督学习方法有效地学习模型
* 可以推理出模型不确定性

缺点：

* 首先学习一个模型，然后构造一个值函数，会有更大的误差。

​		模型学习是一个根据experience的监督学习问题，其中状态转换是概率分布问题，奖励是回归问题。

### 2.2 查表法

​		计算公式如下：
$$
\begin{aligned}
\hat{\mathcal{P}}_{s, s^{\prime}}^{a} &=\frac{1}{N(s, a)} \sum_{t=1}^{T} \mathbf{1}\left(S_{t}, A_{t}, S_{t+1}=s, a, s^{\prime}\right) \\
\hat{\mathcal{R}}_{s}^{a} &=\frac{1}{N(s, a)} \sum_{t=1}^{T} \mathbf{1}\left(S_{t}, A_{t}=s, a\right) R_{t}
\end{aligned}
$$


### 2.3 Planning with a Model

方法：

* Value iteration
* Policy iteration
* Tree search



### 2.4 Sample-Based Planning

​		一个简单但是有效的planning方法，从model中采样experience，然后用Model-free的RL方法求解。

* Monte-Carlo control
* Sarsa
* Q-learning 



## 三、Integrated Architectures  

### 3.1 Dyna算法

Dyna算法思想

* Learn a model from real experience
* Learn and plan value function (and/or policy) from real and simulated experience  



![](https://cdn.mathpix.com/snip/images/kefHr2Oa71WxZx11KNQB0vLeAlan3QVU9Gc8LyEtMKA.original.fullsize.png)



Dyna-Q算法流程：

![](https://cdn.mathpix.com/snip/images/RmPTn14Mco6tQCKJTBHU0fVLzbX-1y6h4h4_nbL_1TA.original.fullsize.png)



## 四、Simulation-Based Search  

### 4.1 Forward Search		

​		不用求解全部的MDP，只需关注从此处开始的MDP分支。在强化学习中，基于模拟的搜索 (simulation-based search) 是一种前向搜索形式，它从当前时刻的状态开始，利用模型来模拟采样，构建一个关注短期未来的前向搜索树，将构建得到的搜  索树作为一个学习资源，使用不基于模型的强化学习方法来寻找当前状态下的最优策略 (图 8.3)。如果使用蒙特卡罗学习方法则称为蒙特卡罗搜索，如果使用 Sarsa 学习方法，则称为 TD 搜索。其中蒙特卡罗搜索又分为简单蒙特卡罗搜索和蒙特卡罗树搜索。  

​		对于一个模型 $M_v$ 和一个一致的模拟过程中使用的策略 $\pi$，简单蒙特卡罗搜索在当前实际状态 $s_t$ 时会针对行为空间中的每一个行为 $a\in A$ 进行 $K$ 次的模拟采样：
$$
\left\{s_{t}, a, R_{t+1}^{k}, S_{t+1}^{k}, A_{t+1}^{k}, \ldots, S_{T}^{k}\right\}_{k=1}^{K} \sim M_{v}, \pi
$$
​		通过计算模拟采样得到的 k 个状态 st 时采取行为 s 的收获的平均值来估算该状态行为对的价值：
$$
Q\left(s_{t}, a\right)=\frac{1}{K} \sum_{k=1}^{K} G_{t}
$$
 		比较行为空间中所有行为 a 的价值，确定当前状态 $s_t$ 下与环境发生实际交互的行为$a_t$：  
$$
a_{t}=\underset{a \in \mathrm{A}}{\operatorname{argmax}} Q\left(s_{t}, a\right)
$$
​		简单蒙特卡罗搜索可以使用基于模拟的采样对当前模拟采样的策略进行评估，得到基于模拟采样的某状态行为对的价值，这个价值的估计同时还与每次采样的 K 值大小有关。在估算行为价值时，关注点在于从当前状态和行为对应的收获，并不关注模拟采样得到的一些中间状态和对应行为的价值。如果同时考虑模拟得到的中间状态和行为的价值，则可以考虑蒙特卡罗树搜索。



### 4.2 蒙特卡罗树搜索  

​		蒙特卡罗树搜索 (Monte-Carlo tree search, MCTS) 在构建当前状态 $s_t$ 的基于模拟的前向搜索时，关注模拟采样中所经历的所有状态及对应的行为，以此构建一个搜索树。利用这颗搜索树不仅可以对当前模拟策略进行评估，还可以改善模拟策略。在使用蒙特卡罗树搜索进行模拟策略评估时，对于个体构建的模型 $M_v$ 和当前的模拟策略$\pi$，在实际当前状态 $s_t$ 时模拟采样出 K 个完整状态序列：  
$$
\left\{s_{t}, A_{t}^{k}, R_{t+1}^{k}, S_{t+1}^{k}, \ldots, S_{T}^{k}\right\}_{k=1}^{K} \sim M_{v}, \pi
$$
​		构建一颗以状态 $s_t$ 为根节点包括所有已访问的状态和行为的搜索树，对树内的每一个状态行为对 $(s; a)$ 使用该状态行为对的平均收获来估算其价值：  
$$
Q(s, a)=\frac{1}{N(s, a)} \sum_{k=1}^{K} \sum_{u=t}^{T} 1\left(S_{u}, A_{u}=s, a\right) G_{u}
$$
​		当搜索结束时，比较当前状态 $s_t$ 下行为空间 A 内的每一个行为的价值，从中选择最大价值对应的行为 $a_t$ 作为当前状态 $s_t$ 时个体与环境实际交互的行为。  

​		比较简单蒙特卡罗搜索和蒙特卡罗树搜索，可以看出两者之间的区别在于前者针对当前状态 $s_t$ 时每一个可能的行为都进行相同数量的采样，而后者则是根据模拟策略进行一定次数的采样。此外，蒙特卡罗树搜索会对模拟采样产生的状态行为对进行计数，并计算其收获，根据这两个数据来计算模拟采样对应的状态行为对价值。比较两者之间的差别可以看出，如果问题的行为空间规模很大，那么使用蒙特卡罗树搜索比简单蒙特卡罗搜索要更实际可行。在蒙特卡罗树搜索中，搜索树的广度和深度是伴随着模拟采样的增多而逐渐增多的。在构建这个搜索树的过程中，搜索树内状态行为对的价值也在不停的更新，利用这些更新的价值信息可以使得在每模拟采样得到一个完整的状态序列后都可以一定程度地改进模拟策略。通常蒙特卡罗树搜索的策略分为两个阶段：  

1. 树内策略 (tree policy)：为当模拟采样得到的状态存在于当前的搜索树中时适用的策略，该策略。树内策略可以使 ϵ-贪婪策略，随着模拟的进行可以得到持续改善；
2. 默认策略 (default policy)：当前状态不在搜索树内时，使用默认策略来完成整个状态序列的采样，并把当前状态纳入到搜索树中。默认策略可以使随机策略或基于某目标价值函数的策略。



​		随着不断地重复模拟，状态行为对的价值将得到持续地得到评估。同时搜索树的深度和广度将得到扩展，策略也不断得到改善。蒙特卡罗树搜索较为抽象，本章暂时介绍到这里，在第十章介绍 AlphaZero 算法时会利用五子棋实例详细讲解蒙特卡罗树搜索的过程细节。  

