import gym
from collections import defaultdict
import numpy as np
from typing import Callable, Tuple,Optional
from tqdm import trange
from env import  *
from policy import *
import matplotlib.pyplot as plt

# MC
def on_policy_mc_control_epsilon_soft(env: gym.Env, num_episodes: int, gamma: float, epsilon: float):
    main_epoch_count_per_round = []
    Tag = "MC ON POLICY EPSILON SOFT"
    for j in trange(10,desc="MC_Epsilon"):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        N = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = create_epsilon_policy(Q, epsilon)
        episode = []
        state = env.reset()
        epoch_counter = []
        for i in range(num_episodes):
            if episode == []:
                action = env.action_space.sample()
            else:
                action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state,action,reward))
            if done or len(episode)==459:
                G=0
                visited_states =[]
                for t in range(len(episode)-1,-1,-1):
                    state,action,reward = episode[t]
                    G = gamma*G+reward
                    pair=(state,action)
                    if pair  not in visited_states:
                        visited_states.append(pair)
                        N[state][action]+=1
                        Q[state][action] += (G-Q[state][action])/N[state][action]
                #done=False
                episode = []
                state = env.reset()
                epoch_counter.append(epoch_counter[-1]+1)
            else:
                state = next_state
                if epoch_counter == []:
                    epoch_counter.append(0)
                else:
                    epoch_counter.append(epoch_counter[-1]+0)
        main_epoch_count_per_round.append(epoch_counter)
    return main_epoch_count_per_round,Tag

def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    main__epoch_count_per_round = []
    Tag="SARSA"
    for time_rep in trange(10,desc="SARSA"):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = create_epsilon_policy(Q, epsilon)
        state = env.reset()
        action = policy(state)
        done = False
        epoch_counter = []
        for i in range(num_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state)
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            if done:
                epoch_counter.append(epoch_counter[-1]+1)
                state = env.reset()
                action = policy(state)
                done = False
            else:
                if (epoch_counter ==[]):
                    epoch_counter.append(0)
                else:
                    epoch_counter.append(epoch_counter[-1]+0)
                state = next_state
                action = next_action
        main__epoch_count_per_round.append(epoch_counter)
    return main__epoch_count_per_round,Tag

# N Step SARSA
def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    main__epoch_count_per_round = []
    Tag="N-Step SARSA"
    for k in trange(10,desc="N-Step-SARSA"):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = create_epsilon_policy(Q,epsilon=epsilon)
        step =0
        n=4
        epoch_counter=[]
        state = env.reset()
        action = policy(state)
        T = float('inf')
        t=0
        rewards=[]
        actions=[action]
        states=[state]
        tau =0
        epoch_counter=[]
        while step<num_steps:
            if t<=T:
                next_state, reward, done, _=env.step(action)
                rewards.append(reward)
                states.append(next_state)
                if done:
                    T = t+1
                else:
                    next_action = policy(next_state)
                    actions.append(next_action)
            tau = t-n+1
            if tau>=0:
                G = sum([gamma**(i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T))])
                if tau + n < T:
                    G += gamma**n * Q[states[tau + n]][actions[tau + n]]

                Q[states[tau]][actions[tau]] += step_size * (G - Q[states[tau]][actions[tau]])    
            if (tau == T-1):
                state = env.reset()
                action = policy(state)
                T = float('inf')
                t=0
                tau =0
                rewards=[0]
                actions=[action]
                states=[state]
                step+=1
                epoch_counter.append(epoch_counter[-1]+1)

            else:
                state = next_state
                action = next_action
                t += 1
                step+=1
                if epoch_counter==[]:
                    epoch_counter.append(0)
                else:
                    epoch_counter.append(epoch_counter[-1]+0)
        main__epoch_count_per_round.append(epoch_counter)
    return main__epoch_count_per_round,Tag

# Expected SARSA
def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    main_epoch_count_per_round=[]
    Tag="Expected_SARSA"
    for time_rep in trange(10,desc="EXP_SARSA"):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = create_epsilon_policy(Q, epsilon)
        state = env.reset()
        action = policy(state)
        done = False
        epoch_counter=[]

        for i in range(num_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state)
            # Expected SARSA update
            expected_q=0
            q_max = np.max(Q[next_state])
            greedy_actions =0
            for i in range(env.action_space.n):
                if Q[next_state][i]==q_max:
                    greedy_actions += 1
            non_greedy_action_probability = epsilon / env.action_space.n
            greedy_action_probability = ((1 - epsilon) / greedy_actions) + non_greedy_action_probability
            for i in range(env.action_space.n):
                if Q[next_state][i] == q_max:
                    expected_q += Q[next_state][i] * greedy_action_probability
                else:
                    expected_q += Q[next_state][i] * non_greedy_action_probability
            Q[state][action] += step_size * (reward + gamma * expected_q - Q[state][action])
            if done:
                epoch_counter.append(epoch_counter[-1]+1)
                state = env.reset()
                action = policy(state)
                done = False
            else:
                if (epoch_counter ==[]):
                    epoch_counter.append(0)
                else:
                    epoch_counter.append(epoch_counter[-1]+0)
                state = next_state
                action = next_action
        main_epoch_count_per_round.append(epoch_counter)
    return main_epoch_count_per_round,Tag

# Q-Learning
def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    main_epoch_count_per_round=[]
    Tag = "Q-Learning"
    for k in trange(10,desc = "Q-learning"):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = create_epsilon_policy(Q, epsilon)
        epoch_counter=[]
        state = env.reset()
        done = False
        for step in range(num_steps):
            action=policy(state)
            next_state, reward, done, _ = env.step(action)
                # Selecting next action greedy policy
            next_action = np.max(Q[next_state])
            max_action = np.argwhere(Q[next_state]==next_action).flatten()
            next_action = np.random.choice(max_action)
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            if done:
                epoch_counter.append(epoch_counter[-1]+1)
                state = env.reset()
                done = False
            else:
                if (epoch_counter ==[]):
                    epoch_counter.append(0)
                else:
                    epoch_counter.append(epoch_counter[-1]+0)
                state = next_state
        main_epoch_count_per_round.append(epoch_counter)
    return main_epoch_count_per_round,Tag
#Plotting 
def plotting(env=gym.Env,num_steps=int(1e4), gamma=1, epsilon=0.1,step_size=0.5 ):
    
    # SARSA
    episode_count,tag=sarsa(env,num_steps,gamma,epsilon,step_size)
    avg_acroos_episode = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_acroos_episode,label=tag)
    std_error1 = 1.96*np.std(avg_acroos_episode)/np.sqrt(10)
    plt.fill_between(np.arange(0,num_steps,1),avg_acroos_episode - std_error1,avg_acroos_episode + std_error1,alpha=0.4)
 
    # EXP_SARSA
    episode_count,tag=exp_sarsa(env,num_steps,gamma,epsilon,step_size)
    avg_acroos_episode = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_acroos_episode,label=tag)
    std_error1 = 1.96*np.std(avg_acroos_episode)/np.sqrt(10)
    plt.fill_between(np.arange(0,num_steps,1),avg_acroos_episode - std_error1,avg_acroos_episode + std_error1,alpha=0.4)
   
    # N-Step Sarsa
    episode_count,tag=nstep_sarsa(env,num_steps,gamma,epsilon,step_size)
    avg_acroos_episode = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_acroos_episode,label=tag)
    std_error1 = 1.96*np.std(avg_acroos_episode)/np.sqrt(10)
    plt.fill_between(np.arange(0,num_steps,1),avg_acroos_episode - std_error1,avg_acroos_episode + std_error1,alpha=0.4)
    
    #MC on Policy
    episode_count,tag=on_policy_mc_control_epsilon_soft(env,num_steps,gamma,epsilon)
    avg_acroos_episode = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_acroos_episode,label=tag)
    std_error1 = 1.96*np.std(avg_acroos_episode)/np.sqrt(10)
    plt.fill_between(np.arange(0,num_steps,1),avg_acroos_episode - std_error1,avg_acroos_episode + std_error1,alpha=0.4)
    
    #Q-Learning
    episode_count,tag=q_learning(env,num_steps,gamma,epsilon,step_size)
    avg_acroos_episode = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_acroos_episode,label=tag)
    std_error1 = 1.96*np.std(avg_acroos_episode)/np.sqrt(10)
    plt.fill_between(np.arange(0,num_steps,1),avg_acroos_episode - std_error1,avg_acroos_episode + std_error1,alpha=0.4)
    plt.legend()
    plt.grid    
    plt.xlabel("Steps")
    plt.ylabel("Episodes")
    plt.title("Algorithm Comparision")
    return plt
# Run This for Question 4 Answer

register_env(1)
env = gym.make('WindyGridWorld-v0',king_move=False,stochiastic=False)
plot=plotting(env,num_steps=8_000)
plt.show()

# Question 5
# Near Optimal Policy Q learning is used
def Q_value_Q_learning(env: gym.Env,num_steps: int,gamma: float,epsilon: float,step_size: float):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    state = env.reset()
    done = False
    for step in range(num_steps):
            action=policy(state)
            next_state, reward, done, _ = env.step(action)
            # Selecting next action greedy policy
            next_action = np.max(Q[next_state])
            max_action = np.argwhere(Q[next_state]==next_action).flatten()
            next_action = np.random.choice(max_action)
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            if done:
                state = env.reset()
                done = False
            else:
                state = next_state
    return Q
register_env(1)
env = gym.make('WindyGridWorld-v0',king_move=False,stochiastic=False)
Q = Q_value_Q_learning(env,int(1e4),1,0.1,0.5)

def episodes(num_eps,optimal_Q):
    episode = []
    for  i in range(num_eps):
        states,actions,rewards =[],[],[]
        state=env.reset()
        policy =create_epsilon_policy(optimal_Q,epsilon=0.1)
        done = False
        while not done:
            action = policy(state)
            states.append(state)
            actions.append(action)
            next_state, reward, done, _ = env.step(action=action)
            rewards.append(reward)
            state=next_state
        episode.append((states,actions,rewards))
    return episode
episodes_50=episodes(num_eps=50,optimal_Q=Q)
episodes_20 = episodes(num_eps=20,optimal_Q=Q)
episodes_1 = episodes(num_eps=1,optimal_Q=Q)
episodes_500 = episodes(num_eps=500,optimal_Q=Q)
def td_prediction(gamma: float, episodes, n=1) -> defaultdict:
    #TODO
    V = defaultdict(float)  # Value function
    evaluation_target =[]
    for episode in episodes:
        states, actions, rewards = episode
        # TD Prediction Update
        T = len(states)
        for t in range(T):
            G_t = sum(gamma ** (i - t - 1) * rewards[i] for i in range(t, min(t + n,len(states))))
            if t + n < T:
                G_t += gamma ** n * V[states[t + n]]
            V[states[t]] += 0.5*(G_t - V[states[t]]) 
            if states[t]==(0,3):
                evaluation_target.append(V[states[t]])
    return V,evaluation_target
"""
print(td_prediction(1,episodes_50,5))
print("\n")
"""
"""
# TD0 Check Function
def td0(env:gym.Env,gamma:float,episodes):
    V = defaultdict(float)
    for episode in episodes:
        states,actions,rewards = episode
        for t in range(len(states)-1):
            V[states[t]] += 0.5 * (rewards[t] + gamma * V[states[t+1]] - V[states[t]])
    return V
print(td0(env,1,episodes_50))
print("\n")
"""
# MC
def monte_carlo_prediction(episodes: List[Tuple[List[int], List[int], List[float]]], gamma: float):
    V = defaultdict(float)
    N = defaultdict(int)
    evaluation_targets = []
    for episode in episodes:
        G = 0
        states,_,rewards = episode
        visited_states=[]
        for t in range(len(states) - 1, -1, -1):
            # TODO Q3a
            # Update V and N here according to first visit MC
            state,reward = states[t],rewards[t]
            G = gamma * G + reward
            if state not in visited_states:
                N[state] = N[state]+1
                V[state] = V[state] + (G-V[state])/N[state]
                visited_states.append(state)
            if state == (0,3):
                evaluation_targets.append(V[state])
    return V,evaluation_targets
evaluation_targets_monte = monte_carlo_prediction(episodes_500,1)[1]
evaluation_targets_TD_4 = td_prediction(gamma=1,episodes=episodes_500,n=1)[1]
plt.hist(evaluation_targets_monte, bins=20, alpha=1, label="MC_500_episode")
plt.legend()
plt.show()



"""
def learning_targets(
    V: defaultdict, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:

    # TODO
    pass
"""