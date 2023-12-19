from scipy.stats import poisson
import numpy as np
from enum import IntEnum
from typing import Tuple


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (0, -1),
        Action.DOWN: (1, 0),
        Action.RIGHT: (0, 1),
        Action.UP: (-1, 0),
    }
    return mapping[action]


class Gridworld5x5:
    """5x5 Gridworld"""

    def __init__(self) -> None:
        """
        State: (x, y) coordinates

        Actions: See class(Action).
        """
        self.rows = 5
        self.cols = 5
        self.state_space = [
            (x, y) for x in range(0, self.rows) for y in range(0, self.cols)
        ]
        self.action_space = len(Action)

        # TODO set the locations of A and B, the next locations, and their rewards
        self.A = (0,1)
        self.A_prime = (4,1)
        self.A_reward = 10.0
        self.B = (0,3)
        self.B_prime = (2,3)
        self.B_reward = 5.0

    def transitions(self, state: Tuple, action: Action) -> Tuple[Tuple[int, int], float]:
        """Get transitions from given (state, action) pair.
        Note that this is the 4-argument transition version p(s',r|s,a).
        This particular environment has deterministic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            next_state: Tuple[int, int]
            reward: float
        """
        next_state = None
        reward = None

        if state == self.A:
            next_state = self.A_prime
            reward = self.A_reward
        elif state == self.B:
            next_state = self.B_prime
            reward = self.B_reward
        else:
        # Calculate the change in (x, y) coordinates based on the chosen action
            dx, dy = actions_to_dxdy(action)

        # Calculate the next state
            x, y = state
            next_x = x + dx
            next_y = y + dy

            # Check if the next step is within the grid boundaries
            if (0 <= next_x < self.rows and 0 <= next_y < self.cols):
                next_state = (next_x, next_y)
                reward = 0.0  # Default reward for non-special states
            else:
            # If the next step is outside the grid boundaries, stay in the current state
                next_state = state
                reward = -1.0  
        return next_state, reward

    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (Action): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        next_state, reward = self.transitions(state, action)
        # TODO compute the expected return
        ret = reward + (gamma * V[next_state[0], next_state[1]])

        return ret
#--------------------------------------------
# Policy Evaluation
#--------------------------------------------
    def  policy_evaluation(self,thetha=1e-3,policy=None,state_value=np.zeros((5,5))):
        main_V = state_value
        if policy == None:
            policy ={}
            for _,state in enumerate(self.state_space):
                policy[state]= (np.ones(self.action_space))/self.action_space
        else:
            policy = policy
        while True:
            terminate = True
            delta = 0
            for s,state in enumerate(self.state_space):
                old_value = main_V[state[0],state[1]]
                new_value = 0

                for a in Action:
                    new_value += policy[state][a]* self.expected_return(main_V,state,action=Action(a),gamma=0.9)
                main_V[state[0],state[1]] = new_value
                #Update the value begins
                #next_state, reward=self.transitions(state,policy[state[0],state[1]])
                #main_V[state[0],state[1]] = self.expected_return(main_V,state,action=policy[state[0],state[1]],gamma=0.9)
                if np.abs(main_V[state[0],state[1]] - old_value) > thetha:
                    terminate = False
            if terminate:
                break
        return main_V
#---------------------------------------------
# Value Iteration
#---------------------------------------------
    def value_iteration_state_value(self,thetha=0.001):
        main_V = np.zeros((5, 5))
        while True:
            terminal = True
            for s,state in enumerate(self.state_space):
                old_v = main_V[state[0],state[1]]
                value_list = []
                for a in Action:
                    value_list.append(self.expected_return(main_V,state,action=Action(a),gamma=0.9))
                main_V[state[0],state[1]] = np.max(value_list)
                if np.abs(main_V[state[0],state[1]] - old_v) > thetha:
                    terminal = False 
            if terminal:
                break
        return main_V
    
    def value_iteration_action(self,optimal_V):

        map={}
        for s,state in enumerate(self.state_space):
            act_lst=[]
            for a in Action:
                expected_return = self.expected_return(optimal_V, state, Action(a), gamma=0.9)
                act_lst.append(np.round(expected_return,2))
            act_lst = np.array(act_lst)
            actions = np.where(act_lst == act_lst.max())[0]
            actions = [Action(a).name for a in actions]
            map[(state[0],state[1])]=actions
        return map
#---------------------------------------------
# Policy Improvement
#---------------------------------------------
    def policy_improvement(self,policy, state_value, gamma):
        policy_stable = True
        for s,state in enumerate(self.state_space):
            old_a = np.argmax(np.array(policy[state]))
            act_lst=[]
            for a in Action:
                act_lst.append(np.round(self.expected_return(state_value,state,Action(a),gamma=gamma),2))                         
            # Select the best new Action.
            act_lst = np.array(act_lst)
            new_a = np.where(act_lst == act_lst.max())[0]
            # check if the policy is stable
            if old_a not in new_a:
                policy_stable = False 
            if  (policy_stable == False):
                i=0
                for i in range(len(policy[state])):
                    if (i == new_a[0]):
                        policy[state][i]=1
                    else:
                        policy[state][i]=0
        return policy_stable,policy
#----------------------------------------------
# Policy Iteration
#----------------------------------------------
    def policy_iteration(self):
        policy ={}
        for _,state in enumerate(self.state_space):
            policy[state]= (np.ones(self.action_space))/self.action_space
        state_value = np.zeros((5,5))
        while True:
            state_value = self.policy_evaluation(policy=policy,state_value=state_value)
            policy_stable,policy=self.policy_improvement(policy=policy,state_value=state_value,gamma=0.9)
            if policy_stable:
                break
        return state_value,policy
C=Gridworld5x5()
state_value,policy = C.policy_iteration()
#Once I get the state_values from policy iteration I am just using the prewritten 
print("These are the Policy Iteration optimal actions","\n")
print(C.value_iteration_action(state_value))
print("This is the optimal state_values received from Policy Iteration ","\n")
print(state_value)









    