# RL Prodigy: Mastering Reinforcement Learning

Embark on a comprehensive RL exploration, spanning from foundational bandit problems to cutting-edge deep RL. Begin with classical algorithms like Epsilon-Greedy and UCB, applied to a 10-armed bandit environment, laying the groundwork for understanding fundamental RL concepts.

The journey advances into on/off-policy Monte Carlo methods, unveiling real-world applications. Shifting focus to TD learning, explore SARSA, Expected SARSA, Q Learning, and n-step SARSA, each shedding light on RL adaptability. A leap into Deep Q Networks (DQN) introduces enhanced strategies, including the Dual DQN (DDQN) model to address overestimation biases, enhancing stability.

Culminating in Proximal Policy Optimization (PPO), witness the convergence of classical RL with cutting-edge deep learning. PPO's robustness marks a milestone in this expedition, synthesizing theoretical foundations with practical insights. Gain valuable perspectives into adaptability, scalability, and challenges across diverse environments, making this project an indispensable resource for enthusiasts and practitioners alike.






## Roadmap

- Multi Arm Bandit (Epsilon Greedy and UCB)

- Add more integrations


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
![4thprob](https://github.com/Hussain7252/RL_Prodigy/assets/124828274/f0c61e5d-d677-4c14-82a9-daf6e74b5719)


