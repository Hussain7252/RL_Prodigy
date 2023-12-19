from env import BanditEnv
from agent import *
from tqdm import trange
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """

    env = BanditEnv(k=k)
    env.reset()

    # TODO
    reward_per_arm = [[] for _ in range(k)]
    for _ in range(num_samples):
        action = np.random.choice(k)
        reward = env.step(action)
        reward_per_arm[action].append(reward)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=reward_per_arm, inner="point")
    plt.xlabel("Action")
    plt.ylabel("Reward Distribution")
    plt.title("Distribution of Sampled Rewards for Each Arm")
    plt.xticks(np.arange(k), np.arange(1, k+1))
    plt.show()


def q6(k: int, trials: int, steps: int, epsilons=[0.0,0.01,0.1],init=0):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(k)
    agents = [EpsilonGreedy(k, init, epsilon=epsilon) for epsilon in epsilons]
    avg_rewards = np.zeros((len(epsilons), trials, steps))
    optimal_actions = np.zeros((len(agents), trials, steps), dtype=int)
    max_reward_lst = []
    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        trial_actions = np.zeros((len(agents), steps), dtype=int)
        highest_reward = np.max(env.means)
        max_reward_lst.append(highest_reward)
        for agent in agents:
            agent.reset()
        # TODO For each trial, perform specified number of steps for each type of agent
        for step in range(steps):
            for i, agent in enumerate(agents):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)
                avg_rewards[i, t, step] = reward
                trial_actions[i, step] = action
        optimal_actions[:, t, :] = trial_actions == np.argmax(env.means)
    return avg_rewards, optimal_actions, max_reward_lst

def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(k)
    agents = [EpsilonGreedy(k,0,0),EpsilonGreedy(k,5,0),EpsilonGreedy(k,0,0.1),EpsilonGreedy(k,5,0),UCB(k,0,2)]
    avg_rewards = np.zeros((len(agents), trials, steps))
    optimal_actions = np.zeros((len(agents), trials, steps), dtype=int)
    max_reward_lst = []
    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        trial_actions = np.zeros((len(agents), steps), dtype=int)
        highest_reward = np.max(env.means)
        max_reward_lst.append(highest_reward)
        for agent in agents:
            agent.reset()
        # TODO For each trial, perform specified number of steps for each type of agent
        for step in range(steps):
            for i, agent in enumerate(agents):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)
                avg_rewards[i, t, step] = reward
                trial_actions[i, step] = action
        optimal_actions[:, t, :] = trial_actions == np.argmax(env.means)
    return avg_rewards, optimal_actions, max_reward_lst


def main():
    # TODO run code for all questions
    
    k = 10
    trials = 2000
    steps = 2000
    epsilons = [0.0,0.01,0.1]
    
    # Question 4
    
    q4(k,steps)

    #Question 6
    
    avg_rewards, optimal_actions, max_reward_lst=q6(k,trials,steps,epsilons)
    #Question 6 Graphs
    optimal_action_fraction = np.mean(optimal_actions, axis=1)
    max_achievable = np.mean(max_reward_lst)
    tot_std_error = 1.96*np.std(max_reward_lst) / np.sqrt(trials)
    upper_bound = max_achievable * np.ones(steps)
    avg_reward_of_all = np.mean(avg_rewards,axis=1)
    for id, epsilon in enumerate(epsilons):
        avg_reward = avg_reward_of_all[id]
        std_error = 1.96*np.std(avg_rewards[id])/np.sqrt(trials)
        plt.plot(np.arange(steps), avg_reward, label=f"Epsilon={epsilon}")
        plt.fill_between(np.arange(steps),avg_reward - std_error,avg_reward + std_error,alpha=0.5)
    plt.plot(np.arange(steps),upper_bound, color='r', linestyle='--', label="Upper Bound")
    plt.fill_between(np.arange(steps),upper_bound - tot_std_error, upper_bound + tot_std_error, alpha = 0.5)
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Over Time with Confidence Bands")
    plt.legend()
    plt.show()

    
    for id, epsilon in enumerate(epsilons):
        opt_action = optimal_action_fraction[id]
        std_error = 1.96*np.std(optimal_actions[id])/np.sqrt(trials)
        plt.plot(np.arange(steps), optimal_action_fraction[id, :]*100, label=f"Epsilon={epsilon}")
        plt.fill_between(np.arange(steps),(opt_action - std_error)*100,(opt_action + std_error)*100, alpha = 0.5)
    plt.xlabel("Step")
    plt.ylabel("% Optimal Action")
    plt.title("Percentage of Times Optimal Action is Selected")
    plt.legend()
    plt.show()

    #Question 7

    trials = 2000
    steps = 1000
    epsilons = [0,0,0.1,0.1]
    Q=[0,5,0,5]
    avg_rewards, optimal_actions, max_reward_lst=q7(k,trials,steps)
    
    optimal_action_fraction = np.mean(optimal_actions, axis=1)
    max_achievable = np.mean(max_reward_lst)
    tot_std_error = 1.96*np.std(max_reward_lst) / np.sqrt(trials)
    upper_bound = max_achievable * np.ones(steps)
    avg_reward_of_all = np.mean(avg_rewards,axis=1)
    for id, epsilon in enumerate(epsilons):
        avg_reward = avg_reward_of_all[id]
        std_error = 1.96*np.std(avg_rewards[id])/np.sqrt(trials)
        plt.plot(np.arange(steps), avg_reward, label=f"Epsilon={epsilon}, Q={Q[id]}")
        plt.fill_between(np.arange(steps),avg_reward - std_error,avg_reward + std_error,alpha=0.5)
    
    avg_reward = avg_reward_of_all[4]
    std_error = std_error = 1.96*np.std(avg_rewards[4])/np.sqrt(trials)
    plt.plot(np.arange(steps), avg_reward, label=f"C={2}")
    plt.fill_between(np.arange(steps),avg_reward - std_error,avg_reward + std_error,alpha=0.5)


    plt.plot(np.arange(steps),upper_bound, color='r', linestyle='--', label="Upper Bound")
    plt.fill_between(np.arange(steps),upper_bound - tot_std_error, upper_bound + tot_std_error, alpha = 0.5)
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Over Time with Confidence Bands")
    plt.legend()
    plt.show()

    for id, epsilon in enumerate(epsilons):
        opt_action = optimal_action_fraction[id]
        std_error = 1.96*np.std(optimal_actions[id])/np.sqrt(trials)
        plt.plot(np.arange(steps), optimal_action_fraction[id, :]*100, label=f"Epsilon={epsilon}")
        plt.fill_between(np.arange(steps),(opt_action - std_error)*100,(opt_action + std_error)*100, alpha = 0.5)
    
    opt_action = optimal_action_fraction[4]
    std_error = 1.96*np.std(optimal_actions[4])/np.sqrt(trials)
    plt.plot(np.arange(steps), optimal_action_fraction[4]*100, label=f"C={2}")
    plt.fill_between(np.arange(steps),(opt_action - std_error)*100,(opt_action + std_error)*100,alpha=0.5)
    
    plt.xlabel("Step")
    plt.ylabel("% Optimal Action")
    plt.title("Percentage of Times Optimal Action is Selected")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()