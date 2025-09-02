import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

NUM_RUNS = 100
NUM_STEPS = 200_000
EVAL_NUM_STEPS = 100_000
K = 10

def choose_action_egreedy(epsilon, estimated_action_values):
        if np.random.rand() < epsilon:
            return np.random.randint(K)
        else:
            return np.argmax(estimated_action_values)

def choose_action_optimistic(estimated_action_values):
    return np.argmax(estimated_action_values)

def choose_action_ucb(estimated_action_values, visit_counts, t, c):
    if np.any(visit_counts == 0):
        zero_idxs = np.flatnonzero(visit_counts == 0)
        return np.random.choice(zero_idxs)
    ucb_values = estimated_action_values + c * np.sqrt(np.log(t) / visit_counts)
    return np.argmax(ucb_values)

def get_gradient_bandit_probs(action_preferences):
    softmax_probs = np.exp(action_preferences) / np.sum(np.exp(action_preferences))
    return softmax_probs

def choose_action_gradient_bandit(probs):
    return np.random.choice(len(probs), p=probs)

def update_action_preferences(action_preferences, average_reward, action, reward, alpha, probs):
    for i in range(K):
        if i == action:
            action_preferences[i] += alpha * (reward - average_reward) * (1 - probs[i])
        else:
            action_preferences[i] -= alpha * (reward - average_reward) * probs[i]

def sample_reward(index, action_values):
    return np.random.randn() + action_values[index]

def run_simulation(epsilon, alpha='avg', strategy='egreedy', initial_av_est=0.0, c=None, gradient_bandit_alpha=0.1):
    
    true_action_values = np.random.randn(K)
    rewards = []

    estimated_action_values = np.array([initial_av_est] * K)
    visit_counts = np.zeros(K)
    action_preferences = np.zeros(K)
    average_reward = 0.0  # For gradient bandit baseline

    for i in range(NUM_STEPS):
        if strategy == 'optimistic':
            action = choose_action_optimistic(estimated_action_values)
        elif strategy == 'egreedy':
            action = choose_action_egreedy(epsilon, estimated_action_values)
        elif strategy == 'ucb':
            action = choose_action_ucb(estimated_action_values, visit_counts, i+1, c)
        elif strategy == 'gradient_bandit':
            action = choose_action_gradient_bandit(get_gradient_bandit_probs(action_preferences))
        reward = sample_reward(action, true_action_values)
        visit_counts[action] += 1
        
        if strategy == 'gradient_bandit':
            probs = get_gradient_bandit_probs(action_preferences)
            update_action_preferences(action_preferences, average_reward, action, reward, gradient_bandit_alpha, probs)
            # Update average reward (incremental update)
            average_reward += (1 / (i + 1)) * (reward - average_reward)
        else:
            # Update estimated action values for non-gradient-bandit strategies
            if alpha == 'avg':
                estimated_action_values[action] += (1 / visit_counts[action]) * (reward - estimated_action_values[action])
            else:
                estimated_action_values[action] += alpha * (reward - estimated_action_values[action])
        
        rewards.append(reward)

        av_increment = np.random.randn(K) * 0.01
        true_action_values += av_increment

    # return average reward over the last EVAL_NUM_STEPS
    return np.mean(rewards[-EVAL_NUM_STEPS:])

def get_simulation_results(epsilon, alpha='avg', strategy='egreedy', initial_av_est=0.0, c=None, gradient_bandit_alpha=0.1):
    with ProcessPoolExecutor() as executor:
        results_iter = executor.map(run_simulation, [epsilon] * NUM_RUNS, [alpha] * NUM_RUNS, [strategy] * NUM_RUNS, [initial_av_est] * NUM_RUNS, [c] * NUM_RUNS, [gradient_bandit_alpha] * NUM_RUNS)
        all_rewards = list(tqdm(results_iter, total=NUM_RUNS))
    return float(np.mean(all_rewards))

if __name__ == "__main__":
    x_axis = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
    x_axis_str = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']
    e_greedy_results = [get_simulation_results(epsilon=epsilon, alpha=.1, strategy='egreedy', initial_av_est=0.0, c=None) for epsilon in x_axis]
    optimistic_results = [get_simulation_results(epsilon=0, alpha=.1, strategy='optimistic', initial_av_est=est, c=None) for est in x_axis]
    ucb_results = [get_simulation_results(epsilon=0, alpha=.1, strategy='ucb', initial_av_est=0.0, c=c) for c in x_axis]
    gradient_bandit_results = [get_simulation_results(epsilon=0, alpha='avg', strategy='gradient_bandit', initial_av_est=0.0, gradient_bandit_alpha=alpha) for alpha in x_axis]
    # plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(x_axis)), e_greedy_results, 'g-', linewidth=1, label='e-greedy')
    plt.plot(range(len(x_axis)), optimistic_results, 'r-', linewidth=1, label='optimistic')
    plt.plot(range(len(x_axis)), ucb_results, 'b-', linewidth=1, label='ucb')
    plt.plot(range(len(x_axis)), gradient_bandit_results, 'y-', linewidth=1, label='gradient bandit')
    plt.xlabel('eps, alpha, c, initial_av_est')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time (averaged over ' + str(NUM_RUNS) + ' runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(x_axis)), x_axis_str)
    plt.show()