import gym
from typing import Callable, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
from policy import create_blackjack_policy, create_epsilon_policy
import matplotlib.pyplot as plt
from algorithms import *

env = gym.make('Blackjack-v1',sab = True)
Q=defaultdict(lambda: np.zeros(env.action_space.n))
policy=create_blackjack_policy(Q)

num_episodes=int(input("Enter number of Episodes: "))

gamma =1.0
V = on_policy_mc_evaluation(env,policy,num_episodes,gamma)

# Plot the state-value function
def plotting(V,num_episodes):
    dealer_usable_ace =[]
    player_sum_usable_ace=[]
    value_usable_ace=[]
    dealer_unusable_ace=[]
    player_sum_unusable_ace=[]
    value_unusable_ace=[]
    for key, vals in V.items():
        
        if (key[2]):
            dealer_usable_ace.append(key[1])
            player_sum_usable_ace.append(key[0])
            value_usable_ace.append(vals)
        if (key[2]==False):
            dealer_unusable_ace.append(key[1])
            player_sum_unusable_ace.append(key[0])
            value_unusable_ace.append(vals)
    fig = plt.figure(figsize=(20,20))
    # Create two Different Plots one for usable ace and another non-usable ace
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        # Plot the data in their respective plots
    ax1.plot_trisurf(dealer_usable_ace, player_sum_usable_ace, value_usable_ace, cmap='viridis', edgecolor='black')
    ax2.plot_trisurf(dealer_unusable_ace, player_sum_unusable_ace, value_unusable_ace, cmap='viridis', edgecolor='black')
    ax1.set_xlim3d(1, 10, 1)
    ax1.set_ylim3d(12, 21, 1)
    ax1.set_zlim3d(-1, 1)
    ax2.set_xlim3d(1, 10, 1)
    ax2.set_ylim3d(12, 21, 1)
    ax2.set_zlim3d(-1, 1)

    ax1.set_title("Useable Ace", size=20)
    ax2.set_title("No Useable Ace", size=20)

    ax1.set_xlabel('Dealer showing', size=12)
    ax1.set_ylabel('Player sum', size=12)
    ax1.set_zlabel('Value', size=12)

    ax2.set_xlabel('Dealer showing', size=12)
    ax2.set_ylabel('Player sum', size=12)
    ax2.set_zlabel('Value', size=12)
    num_episodes = str(num_episodes)
    fig.suptitle('Episodes: ' + num_episodes,size=25)
    plt.show()

plotting(V,num_episodes)