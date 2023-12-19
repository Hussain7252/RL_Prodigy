import gym
from typing import Callable, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
from policy import create_blackjack_policy, create_epsilon_policy
import matplotlib.pyplot as plt
from env import  *

def generate_episode(env: gym.Env, policy: Callable, es: bool = False,terminate: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    if isinstance(state[0],tuple):
        state = state[0]
    count = 0
    if terminate:
        while True:
            if es and len(episode) == 0:
                action = env.action_space.sample()
            else:
                action = Action(policy(state))

            next_state, reward, done, _ ,_= env.step(action)
            episode.append((state, action, reward))
            count += 1
            if done or count == 459:
                break
            state = next_state
    else:
        while True:
            if es and len(episode) == 0:
                action = env.action_space.sample()
            else:
                action = policy(state)

            next_state, reward, done, _ ,_= env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
    return episode


def on_policy_mc_evaluation(
    env: gym.Env,
    policy: Callable,
    num_episodes: int,
    gamma: float,
) -> defaultdict:
 
    # We use defaultdicts here for both V and N for convenience. The states will be the keys.
    V = defaultdict(float)
    N = defaultdict(int)

    for _ in trange(num_episodes, desc="Episode"):
        episode = generate_episode(env, policy)
        G = 0
        visited_states=[]
        for t in range(len(episode) - 1, -1, -1):
            # TODO Q3a
            # Update V and N here according to first visit MC
            state,action,reward = episode[t]
            G = gamma * G + reward
            if state not in visited_states:
                N[state] = N[state]+1
                V[state] = V[state] + (G-V[state])/N[state]
                visited_states.append(state)
    return V

def on_policy_mc_control_es(
    env: gym.Env, num_episodes: int, gamma: float
) -> Tuple[defaultdict, Callable]:

    # We use defaultdicts here for both Q and N for convenience. The states will be the keys and the values will be numpy arrays with length = num actions
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    #States are traversed used argmax(Q) oe else use the default policy
    policy = create_blackjack_policy(Q)
    
    for _ in trange(num_episodes, desc="Episode"):
        # TODO Q3b
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        visited_states = []
        episode = generate_episode(env, policy, es=True)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            pair=(state,action)
            if pair not in visited_states:
                N[state][action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state][action]
                visited_states.append(pair)
                policy = create_blackjack_policy(Q) 
    return Q, policy

def on_policy_mc_control_epsilon_soft(env: gym.Env, num_episodes: int, gamma: float, epsilon: float):


    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    returns = np.zeros(num_episodes)
    for i in trange(num_episodes, desc="Episode"):
        episode = generate_episode(env, policy, terminate=True)
        G = 0
        visited_states = []
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            pair = (state, action)
            if pair not in visited_states:
                visited_states.append(pair)
                N[state][action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state][action]
        returns[i] = G
    return returns
