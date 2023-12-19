import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Sequence


def argmax(arr: Sequence[float]) -> int:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
    Args:
        arr: sequence of values
    """
    best_actions = np.where(arr == np.max(arr))[0]
    Q_highest=np.random.choice(best_actions)
    return Q_highest


class BanditAgent(ABC):
    def __init__(self, k: int, init: int, step_size: float) -> None:
        """Abstract bandit agent class

        Implements common functions for both epsilon greedy and UCB

        Args:
            k (int): number of arms
            init (init): initial value of Q-values
            step_size (float): step size
        """
        self.k = k
        self.init = init
        self.step_size = step_size

        # Q-values for each arm
        self.Q = self.init * np.ones(self.k,dtype=np.float32)
        # Number of times each arm was pulled
        self.N = np.zeros(self.k,dtype=np.int8)
        # Current total number of steps
        self.t = 0

    def reset(self) -> None:
        """Initialize or reset Q-values and counts

        This method should be called after __init__() at least once
        """
        self.Q = self.init * np.ones(self.k, dtype=np.float32)
        self.N = np.zeros(self.k, dtype=int)
        self.t = 0

    @abstractmethod
    def choose_action(self) -> int:
        """Choose which arm to pull"""
        raise NotImplementedError

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """Update Q-values and N after observing reward.

        Args:
            action (int): index of pulled arm
            reward (float): reward obtained for pulling arm
        """
        raise NotImplementedError


class EpsilonGreedy(BanditAgent):
    def __init__(
        self, k: int, init: int, epsilon: float, step_size: Optional[float] = None
    ) -> None:
        """Epsilon greedy bandit agent

        Args:
            k (int): number of arms
            init (init): initial value of Q-values
            epsilon (float): random action probability
            step_size (float or None): step size. If None, then it is equal to 1 / N_t (dynamic step size)
        """
        super().__init__(k, init, step_size)
        self.epsilon = epsilon

    def choose_action(self):
        """Choose which arm to pull

        With probability 1 - epsilon, choose the best action (break ties arbitrarily, use argmax() from above). With probability epsilon, choose a random action.
        """
        if (np.random.rand() < self.epsilon):
            action = np.random.choice(self.k)
            return action
        else:
            action = argmax(self.Q)
        return action

    def update(self, action: int, reward: float) -> None:
        """Update Q-values and N after observing reward.

        Args:
            action (int): index of pulled arm
            reward (float): reward obtained for pulling arm
        """
        self.t += 1

        # TODO update self.N
        self.N[action] += 1
        # TODO update self.Q
        # If step_size is given (static step size)
        if self.step_size is not None: 
            self.Q[action] = self.Q[action] + (self.step_size *(reward - self.Q[action]))
        # If step_size is dynamic (step_size = 1 / N(a))
        else:
            if self.N[action] == 0:
                step_size = 0
            else: 
                step_size = 1.0 / self.N[action]
            self.Q[action] = self.Q[action] + (step_size * (reward - self.Q[action]))


class UCB(BanditAgent):
    def __init__(self, k: int, init: int, c: float, step_size: Optional[float]=None) -> None:
        """Epsilon greedy bandit agent

        Args:
            k (int): number of arms
            init (init): initial value of Q-values
            c (float): UCB constant that controls degree of exploration
            step_size (float): step size (use constant step size in case of UCB)
        """
        super().__init__(k, init, step_size)
        self.c = c

    def choose_action(self):
        """Choose which arm to pull

        Use UCB action selection. Be sure to consider the case when N_t = 0 and break ties randomly (use argmax() from above)
        """
        randomAction = np.random.choice(self.k,1)
        randomAction = int(list(randomAction)[0])
        if self.t <=1:
            action = randomAction
        else:
            ucb_values = self.Q + self.c * np.sqrt(np.log(self.t + 1) / (self.N + 1e-5))
        # Handle the case when N_t = 0 by setting UCB values to a large value (encourage exploration)
            ucb_values[self.N == 0] = float('inf')
            action=argmax(ucb_values)
        return action

    def update(self, action: int, reward: float) -> None:
        """Update Q-values and N after observing reward.

        Args:
            action (int): index of pulled arm
            reward (float): reward obtained for pulling arm
        """
        self.t += 1

        # TODO update self.N
        self.N[action] += 1
        # TODO update self.Q
        if self.step_size is not None: 
            self.Q[action] = self.Q[action] + (self.step_size *(reward - self.Q[action]))
        # If step_size is dynamic (step_size = 1 / N(a))
        else:
            if self.N[action] == 0:
                step_size = 0
            else: 
                step_size = 1.0 / self.N[action]
            self.Q[action] = self.Q[action] + (step_size * (reward - self.Q[action]))

