# Lecture 3: Planning by Dynamic Programming  动态规划寻找最优策略  

[TOC]

## 1. Introduction 介绍

​		**Dynamic**：问题是时序或者顺序的。

​		**Programming**：规划是指在已知环境动力学（环境的全部状态）的基础上寻找最优策略和最优价值函数

​		**动态规划的思想**：是将复杂的问题分解为求解子问题，通过求解子问题得到整个问题的解，在解决子问题的时候，其结果通常需要存储起来被用来解决后续复杂问题。 

​		当问题具有下列两个性质时，通常可以考虑使用动态规划来求解：第一个性质是一个复杂问题的最优解由数个小问题的最优解构成，可以通过寻找子问题的最优解来得到复杂问题的最优解；第二个性质是子问题在复杂问题内重复出现，使得子问题的解可以被存储起来重复利用。  

​		预测：求解基于某一策略的价值函数  。

​		控制：求解最优价值函数和最优策略  。



## 2. Policy Evaluation 策略评估

​		**策略评估 (policy evaluation) **指计算给定策略下状态价值函数的过程，状态价值函数迭代公式：

$$
v_{k+1}(s)=\sum_{a \in A} \pi(a | s)\left(R_{s}^{a}+\gamma \sum_{s^{\prime} \in S} P_{s s^{\prime}}^{a} v_{k}\left(s^{\prime}\right)\right)
$$


## 3. Policy Iteration 策略迭代

​		依据新的策略$\pi'$会得到一个新的价值函数，并产生新的贪婪策略，如此重复循环迭代将最终得到最优价值函数$v^*$ 和最优策略 $\pi^*$。策略在循环迭代中得到更新改善的过程称为**策略迭代**。

![](https://cdn.mathpix.com/snip/images/c25ZEZGONnkh10uvYoGKn2rIMwyLM9EcQ7e6Ammyjow.original.fullsize.png)  

​		基于贪婪策略的迭代将收敛于最优策略和最有状态价值函数的证明  ：

$$
\pi^{\prime}(s)=\underset{a \in A}{\operatorname{argmax}} q_{\pi}(s, a)
$$


​		假如个体在与环境交互的仅下一步采取该贪婪策略产生的行为，而在后续步骤仍采取基于原策略产生的行为，那么下面的（不）等式成立  。（确定性策略）

$$
q_{\pi}\left(s, \pi^{\prime}(s)\right)=\max _{a \in A} q_{\pi}(s, a) \geq q_{\pi}(s, \pi(s))=v_{\pi}(s)
$$
​		如果后续状态均使用贪婪策略：

$$
\begin{aligned}
v_{\pi}(s) & \leq q_{\pi}\left(s, \pi^{\prime}(s)\right)=\mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) | S_{t}=s\right] \\
& \leq \mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma q_{\pi}\left(S_{t+1}, \pi^{\prime}\left(S_{t+1}\right)\right) | S_{t}=s\right] \\
& \leq \mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} q_{\pi}\left(S_{t+2}, \pi^{\prime}\left(S_{t+2}\right)\right) | S_{t}=s\right] \\
& \leq \mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma R_{t+2}+\ldots | S_{t}=s\right]=v_{\pi^{\prime}}(s)
\end{aligned}
$$
​		如果在某一个迭代周期内，状态价值函数不再改善，即：  

$$
q_{\pi}\left(s, \pi^{\prime}(s)\right)=\max _{a \in A} q_{\pi}(s, a)=q_{\pi}(s, \pi(s))=v_{\pi}(s)
$$
​		那么就满足了贝尔曼最优方程的描述：  

$$
v_{\pi}=\max _{a \in A} q_{\pi}(s, a)
$$


## 4. Value Iteration 价值迭代

​		一个策略能够获得某状态 s 的最优价值当且仅当该策略也同时获得状态 s 所有可能的后续状态 s′ 的最优价值。一个状态的最优价值可以由其后续状态的最优价值通过前一章所述的贝尔曼最优方程来计算：  

$$
v_{*}(s)=\max _{a \in A}\left(R_{s}^{a}+\gamma \sum_{s^{\prime} \in S} P_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)\right)
$$




## 5. Extensions to Dynamic Programming 

​		异步动态规划算法  ：前文所述的系列算法均为同步动态规划算法，它表示所有的状态更新是同步的。与之对应的还有异步动态规划算法。在这些算法中，每一次迭代并不对所有状态的价值进行更新，而是依据一定的原则有选择性的更新部分状态的价值，这种算法能显著的节约计算资源，并且只要所有状态能够得到持续的被访问更新，那么也能确保算法收敛至最优解。比较常用的异步动态规划思想有：原位动态规划、优先级动态规划、和实时动态规划等。  



