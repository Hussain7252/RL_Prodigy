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

num_episodes=int(input("Enter number of Episodes: "))
gamma =1.0
Q_opt,pol =  on_policy_mc_control_es(env,num_episodes,gamma)
V = on_policy_mc_evaluation(env,pol,num_episodes,gamma)

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

# Plotting the Optimal Policy
ua_p = {k:v for k,v in Q_opt.items() if k[2]}
nua_p = {k:v for k,v in Q_opt.items() if not k[2]}

def plot_policy_1(policy, ua, title):
    car_num_lot1 = list(np.linspace(start=21, stop=12, num=10, dtype=int))
    car_num_lot2 = list(np.linspace(start=1, stop=11, num=10, dtype=int))
    
    p = np.zeros((len(car_num_lot1),len(car_num_lot2)))

    for i in range(10):
        for j in range(1, 11):
            try:
                p[i,j-1] = np.argmax(policy[(21-i,j,ua)])
            except KeyError:
                print(i-12, j-1)
                continue

    
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(p, cmap="Set1")
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(car_num_lot2)), labels=car_num_lot2)
    ax.set_yticks(np.arange(len(car_num_lot1)), labels=car_num_lot1)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(car_num_lot1)):
        for j in range(len(car_num_lot2)):
            text = ax.text(j, i,int(p[i, j]),
                           ha="center", va="center", color="w")
    
    ax.set_xlabel('Dealer showing', size=12)
    ax.set_ylabel('Player sum', size=12)
    ax.set_title(title, size=20)
    fig.tight_layout()
    plt.show()
plot_policy_1(ua_p, True, "Useable Ace")
plot_policy_1(nua_p, False, "Non Useable Ace")