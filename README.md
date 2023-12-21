# RL Prodigy: Mastering Reinforcement Learning

Embark on a comprehensive RL exploration, spanning from foundational bandit problems to cutting-edge deep RL. Begin with classical algorithms like Epsilon-Greedy and UCB, applied to a 10-armed bandit environment, laying the groundwork for understanding fundamental RL concepts.

The journey advances into on Monte Carlo methods, unveiling real-world applications. Shifting focus to TD learning, explore SARSA, Expected SARSA, Q Learning, and n-step SARSA, each shedding light on RL adaptability. A leap into Deep Q Networks (DQN) introduces enhanced strategies, including the Dual DQN (DDQN) model to address overestimation biases, enhancing stability.

Culminating in Proximal Policy Optimization (PPO), witness the convergence of classical RL with cutting-edge deep learning. PPO's robustness marks a milestone in this expedition, synthesizing theoretical foundations with practical insights. Gain valuable perspectives into adaptability, scalability, and challenges across diverse environments, making this project an indispensable resource for enthusiasts and practitioners alike.






## Roadmap

- Multi Arm Bandit (Epsilon Greedy and UCB)
- Dynamic Programming
- Monte Carlo methods (MC evaluation, MC exploring starts, MC epsilon greedy) on Black jack and Four room evironment
- On Policy Monte Carlo Control, SARSA, Expected SARSA, N-Step SARSA and Q learning on a custom windy gridworld environment
- DQN on Cartpole, Mountain Car, Acrobot and Lunar Lander
- Dueling DDQN, DQN and PPO on Lunar Lander environment

## Section Details

### Multi Arm Bandit

#### Code structure

- The multi-armed bandit environment is defined in `env.py`
- Epsilon-greedy/UCB agents are given in `agents.py`. Both agents inherit the abstract class `BanditAgent`.
- To break ties randomly, we define our own argmax function in `agents.py`.
- The main file to run code is `main.py`, with functions for the environment, epsilon greedy and UCB.

#### Implementation Details
##### Environment:

The 10-armed bandit environment is designed to simulate the multi-arm bandit problem. The true reward for each bandit is sampled from a standard normal distribution, and the observed rewards are generated with a unit variance normal distribution around the mean reward of each arm.

##### Algorithms:
1.) The Epsilon-Greedy algorithm is a simple exploration-exploitation strategy. It selects the best-known action with probability 
1−ϵ and explores with probability 
ϵ, where 
ϵ is a small positive value. This allows the algorithm to strike a balance between exploiting the current knowledge and exploring new actions.

2.) Upper-Confidence Bound action selection uses uncertainty in the action-value estimates for balancing exploration and exploitation. Since there is inherent uncertainty in the accuracy of the action-value estimates when we use a sampled set of rewards thus UCB uses uncertainty in the estimates to drive exploration.

#### Run Locally
clone the project 
``` bash
git clone git@github.com:Hussain7252/ReinforcementLearning_Odessey.git
```
```bash
cd Multi-Arm_Bandit
```
```bash
pip3 install -r requirements.txt
```
```bash
python3 main.py
```
#### Results:
<p align="center">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/bandit_env.png" width="400" height="300" alt="10 Arm Bandit">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/7rewards.png" width="400" height="300" alt="Epsilon greedy and UCB rewards">
</p>

### Dynamic Programming
#### Code structure

- The 5*5 grid world environment and the implementation of Policy Evaluation, Policy Improvement, Policy iteration and value iteration is defined in `env.py`

#### Implementation Details
##### Environment:

A 5*5 empty grid world evironment is defined having a 0 reward for all the grid cells except the special grid cells where the agent gets a reward of 10 and 5 respectively on taking any action in that cell. If an action is taken that causes the agent to collide with the wall it incurs a cost of -1. 

##### Algorithms:
1. Policy Evaluation: It involves estimating the expected return for each state under the specified policy. This is typically done through iterative methods, such as the Bellman Expectation Equation, which updates the value function until convergence.
2. Policy Improvement: By iteratively modifying the policy to choose actions that lead to higher expected returns, policy improvement aims to refine the decision-making strategy. The Greedy Policy Improvement theorem suggests selecting actions with the highest estimated values.
3. Policy Iteration:  It alternates between policy evaluation (estimating the value function for the current policy) and policy improvement (modifying the policy to be more greedy with respect to the current value function). This process continues until the policy converges to an optimal one.
4. Value Iteration:  It involves iteratively updating the value function using the Bellman Optimality Equation until convergence. The resulting value function represents the optimal expected return for each state. The optimal policy can then be derived by selecting actions that maximize the value.

#### Run Locally
clone the project 
``` bash
git clone git@github.com:Hussain7252/ReinforcementLearning_Odessey.git
```
```bash
cd Dynamic_Programming
```
```bash
pip3 install -r requirements.txt
```
```bash
python3 env.py
```

### MC control, SARSA, Expected SARSA, N-Step SARSA and Q Learning on windy grid world environment

#### Code Structure
- The general, stochiastic and windy grid world with diagonal actions is implemented in the `env.py` .
- The epsilon greedy policy is defined in `policy.py`
- MC control with epsilon greedy policy, SARSA, Expected SARSA, N-Step SARSA and Q learning is available in `algorithms.py`.
- The main file to run codes are `algorithms.py`. 

#### Run Locally
clone the project 
``` bash
git clone git@github.com:Hussain7252/ReinforcementLearning_Odessey.git
```
```bash
cd TD-Learning
```
```bash
pip3 install --file requirements.txt
```
Run these files individually
```bash
python3 algorithms.py
```
#### Results
<p align="center">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/Windy_grid.png" width="400" height="300" alt="Windy-Grid World">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/kings_move.png" width="400" height="300" alt="Windy-Grid with kings move">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/stochiasticwind_kings.png" width="400" height="300" alt="Windy-Grid king with stochastic wind">
</p>

### Monte Carlo on Blackjack and FourRoom environment

#### Code Structure
- The blackjack environment from gym (version 0.22.0) is utalized.
- The four room environment is defined in `env.py`
- The blackjack policy and epsilon greedy policy which is common for the black jack and Four room environment is defined in `policy.py`
- MC evaluation, MC control with exploring starts and MC control with epsilon greedy policy are given in `algorithms.py`.
- The main file to run codes are `3a.py` for blackjack MC evaluation, `3b.py` for MC control with exploring starts and `fourroom.py` for MC control with epsilon greedy policy applied on four room environment. 

#### Run Locally
clone the project 
``` bash
git clone git@github.com:Hussain7252/ReinforcementLearning_Odessey.git
```
```bash
cd MonteCarlo
```
```bash
pip3 install --file requirements.txt
```
Run these files individually
```bash
python3 3a.py
python3 3b.py
python3 fourroom.py
```
#### Results
<p align="center">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/3a_500000.png" width="400" height="300" alt="State-Value">
</p>
<p align="center">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/3b_unusable_500000.png" width="400" height="300" alt="Policy_unusableace">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/3b_usable_500000.png" width="400" height="300" alt="Policy_usableace">
</p>
<p align="center">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/question4.png" width="400" height="300" alt="average_rewards_1000episodes">
</p>


### DQN on Cart Pole, Mountain Car, Acrobot and Lunar Lander
#### Code structure

- A jupyter notebook with the implementation of DQN is available in the DQN folder

#### Implementation Details

##### Algorithms:
The DQN utilizes a 3 layered Neural Network to estimate the Q values with the target network being updated every 10000 steps. Epsilon greedy action selection policy is implemented with an exponentially decreasing epsilon. The network is trained for 1500000 steps for all the environments.

#### Run Locally
clone the project 
``` bash
git clone git@github.com:Hussain7252/ReinforcementLearning_Odessey.git
```
```bash
cd DQN
```
In the anaconda prompt
```bash
conda create --name myenv python=3.9
conda activate myenv
```
```bash
conda install --file requirements.txt
```
Launch jypter notebook
```bash
Run the notebook_dqn.ipynb step by step
```

#### Results
<p align="center">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/cartpole_DQN.png" width="400" height="300" alt="DQN_cartpole">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/mountaincar.png" width="400" height="300" alt="DQN_mountaincar">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/acrobot.png" width="400" height="300" alt="DQN_acrobot">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/LunarLander.png" width="400" height="300" alt="DQN_LunarLander">
</p>

### Exploring the Effectiveness of DQN, Dueling DDQN and PPO algorithms on Luunar Lander
#### Code structure

- A jupyter notebook with the implementation of DQN, Dueling DDQN and PPO is available in the PolicybasedVSvaluebased folder

#### Documentation

- [Exploring the Effectiveness of DQN, Dueling DDQN and PPO Algorithms on the Lunar Lander, Hussain Kanchwala](https://github.com/Hussain7252/RL_Prodigy/blob/main/PolicybasedVSvaluebased/Report.pdf)

#### Implementation Details
##### Algorithms:
1. DQN:
   - The idea behind DQN is the use of  neural network to approximate the Q function.
   - To overcome the instability and divergence issue associated with training Deep Neural Network experience replay is used where past transitions are stored in replay buffer andrandomly sampled during training. This helps break temporal correlations and improves stability.
   - Also in order to maintain stability for training more efficiently and preventing oscillations and divergence during training Actor and Target networks are used. So the Acotr network is the main Neural Network that is continuously updated in order to traverse and the target netowrk is used to find the action and the value of it that has to be taken in the next state in order to calculate the target Q value and update the Actor network by finding the loss between Q value given by Actor and Q value generated by the target.
   - Please look at the jupyter notebook for further details on implementation.
2. Dueling DDQN:
   - The DQN causes overestimation of Q values which leads to a suboptimal policy being learned. In order to overcome this we make a change in how we are finding the target values in DQN. Recall in DQN the target network is used both to select the action and  well as to estimate the value of that action in the next state, so in DDQN the the action selection and the value estimation of this next state is done by two different neural networks. You can either use your Actor network to select the action and target network to estimate the value or vice versa.
   -  Dueling DDQN is an extension of the original DDQN that introduces a dueling architecture for the neural network. Instead of having a single stream of output representing the Q-values for each action, the network is split into two streams: one for estimating the state value (V(s)) and another for estimating the advantages of each action (A(a)).
   -  Please look at the respective jupyter notebook for further details on implementation.

3. PPO
   - PPO is an policy optimization algorithm that directly optimizes the policy where as DQN and DDQN are value based methods.
   - It optimizes a surrogative objective function that ensures policy doesnot deviate too far from the previous policy given an update.
   - The objective function encourages policy updates that increase the probability of actions that have higher returns and decrease the probability of actions with lower returns. The clipping term helps prevent overly large policy updates.
   - Please look at the respective jupyter notebook for further details on implementation
   - [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO,Logan et. al.](https://doi.org/10.48550/arXiv.2005.12729)
  
#### Run Locally
clone the project 
``` bash
git clone git@github.com:Hussain7252/ReinforcementLearning_Odessey.git
```
```bash
cd PolicybasedVSvaluebased
```
In the anaconda prompt
```bash
conda create --name myenv python=3.9
conda activate myenv
```
```bash
conda install --file requirements.txt
```
Launch jypter notebook
```bash
Run the DQN.ipynb for DQN on Lunar Lander
Run the deulingdoubleql.ipynb for  Dueling DDQN on Lunar Lander
Run the PPO.ipynb for PPO on Lunar Lander
```

#### Results
<p align="center">
  <img src="https://github.com/Hussain7252/RL_Prodigy/blob/main/media/Final%20plots%20comparision.png">
</p>

#### Acknowledgements:
- [ROBOT LEARNING, edited by Jonathan H. Connell and Sridhar Mahadevan, Kluwer, Boston, 1993/1997, xii 240 pp., ISBN 0-7923-9365-1 (Hardback, 218.00 Guilders, $120.00, £89.95). Robotica. 1999;17(2):229-235. doi:10.1017/S0263574799271172](https://doi.org/10.1017/S0263574799271172)
- [OpenAI](https://openai.com/research/openai-baselines-ppo)
