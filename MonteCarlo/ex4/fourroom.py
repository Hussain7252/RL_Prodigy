import gym
from typing import Callable, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
from policy import create_blackjack_policy, create_epsilon_policy
import matplotlib.pyplot as plt
from algorithms import *
from env import *

goal_pos = (10, 10)
env = FourRoomsEnv(goal_pos=goal_pos)
n_trials = 10
n_eps = 10000
gamma = 0.99


trial_returns_1 = np.zeros((n_trials, n_eps))
trial_returns_2 = np.zeros((n_trials, n_eps))
trial_returns_3 = np.zeros((n_trials, n_eps))
for t in range(n_trials):
    r=on_policy_mc_control_epsilon_soft(env,num_episodes=n_eps,gamma=gamma,epsilon=0.1)
    for idx,val in enumerate(r):
        trial_returns_1[t,idx] = val
average1 = np.mean(trial_returns_1,axis=0)
std_error1 = 1.96*np.std(average1)/np.sqrt(n_trials)
for t in range(n_trials):
    r=on_policy_mc_control_epsilon_soft(env,num_episodes=n_eps,gamma=gamma,epsilon=0.02)
    for idx,val in enumerate(r):
        trial_returns_2[t,idx] = val
average2 = np.mean(trial_returns_2,axis=0)
std_error2 = 1.96*np.std(average2)/np.sqrt(n_trials)
for t in range(n_trials):
    r=on_policy_mc_control_epsilon_soft(env,num_episodes=n_eps,gamma=gamma,epsilon=0)
    for idx,val in enumerate(r):
        trial_returns_3[t,idx] = val
average3 = np.mean(trial_returns_3,axis=0)
std_error3 = 1.96*np.std(average3)/np.sqrt(n_trials)

plt.plot(average1,label='epsilon 0.1')
plt.plot(average2,label='epsilon 0.02')
plt.plot(average3,label="epsilon 0")
plt.fill_between(np.arange(n_eps),average1 - std_error1,average1 + std_error1,alpha=0.65)
plt.fill_between(np.arange(n_eps),average2 - std_error2,average2 + std_error2,alpha=0.65)
plt.fill_between(np.arange(n_eps),average3 - std_error3,average3 + std_error3,alpha=0.65)
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Average Reward Over Time with Confidence Bands")
plt.legend()
plt.show()
