import numpy as np
from collections import defaultdict
from typing import Callable, Tuple
import gym
from enum import IntEnum
from env import *
# Default Policy
def default_blackjack_policy(state: Tuple[int, int, bool]) -> int:
    """default_blackjack_policy.

    Returns sticking on 20 or 21 and hit otherwise

    Args:
        state: the current state
    """
    if state[0] in [20, 21]:
        return 0
    else:
        return 1

# Deterministic Policy or Predefined Policy 
def create_blackjack_policy(Q: defaultdict) -> Callable:
    """Creates an initial blackjack policy from default_blackjack_policy but updates policy using Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """

    def get_action(state: Tuple) -> int:
        # If state was never seen before, use initial blackjack policy
        if state not in Q.keys():
            return default_blackjack_policy(state)
        else:
            # Choose deterministic greedy action
            chosen_action = np.argmax(Q[state]).item()
            return chosen_action

    return get_action

def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """

    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        # TODO
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        if np.random.random() < epsilon:
            action = np.random.choice(num_actions)
        else:
            action = np.max(Q[state])
            max_action = np.argwhere(Q[state]==action).flatten()
            action = np.random.choice(max_action)
        return action
    return get_action

