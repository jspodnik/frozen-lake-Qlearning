import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Q-learning settings from part G
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.05
EPISODES = 2000
EVAL_EPISODES = 100
SEED = 42
WINDOW_SIZE = 100

# noise levels to test
NOISE_LEVELS = [0.0, 0.10, 0.25]

os.makedirs("results", exist_ok=True)

# epsilon-greedy action selection
def choose_action(Q, state, rng, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))

    best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
    return int(rng.choice(best_actions))

# stochastically replace actions due to noise
def apply_action_noise(action, n_actions, noise, rng):
    if rng.random() < noise:
        alternatives = [a for a in range(n_actions) if a != action]
        return int(rng.choice(alternatives))
    return action


def train_q_learning(noise):
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    episode_rewards = []
    rng = np.random.default_rng(SEED + int(noise * 1000))

    for episode in range(EPISODES):
        state, info = env.reset(seed=SEED+episode)
        done = False
        total_reward = 0

        while not done:
            action = choose_action(Q, state, rng, EPSILON)
            actual_action = apply_action_noise(action, n_actions, noise, rng)
            
            next_state, reward, terminated, truncated, info = (env.step(actual_action))

            if terminated:
                target = reward
            else:
                target = reward + GAMMA * np.max(Q[next_state])

            Q[state, action] += ALPHA * (target - Q[state, action])
            total_reward += reward
            state = next_state
            done = terminated or truncated

        episode_rewards.append(total_reward)

    env.close()
    return Q, np.array(episode_rewards)

# evaluate learned policy
def evaluate_policy(Q, noise, seed_offset):
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False)
    n_actions = env.action_space.n
    rng = np.random.default_rng(SEED + seed_offset)
    rewards = []
    steps_list = []

    for episode in range(EVAL_EPISODES):
        state, info = env.reset(seed=SEED + seed_offset + episode)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # no exploration
            action = int(np.argmax(Q[state]))

            # implement environmental noise 
            actual_action = apply_action_noise(action, n_actions, noise, rng)

            state, reward, terminated, truncated, info = (env.step(actual_action))
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        steps_list.append(steps)

    env.close()

    return {
        'mean_reward': float(np.mean(rewards)),
        'success_rate': float(np.mean(rewards)),
        'mean_steps': float(np.mean(steps_list))
    }



# ------- Experiment 1: train separately at each noise level -------
trained_policies = {}
training_rewards = {}
matched_results = {}
for noise in NOISE_LEVELS:
    print(f'\nTraining with action noise p={noise:.2f}...')
    
    Q, rewards = train_q_learning(noise)
    trained_policies[noise] = Q
    training_rewards[noise] = rewards

    matched_results[noise] = evaluate_policy(Q, noise, seed_offset=20000 + int(noise * 1000))
    result = matched_results[noise]
    print(f'Evaluation noise: {noise:.2f}')
    print(f'Mean reward: {result["mean_reward"]:.3f}')
    print(f'Success rate: {result["success_rate"]:.1%}')
    print(f'Average steps: {result["mean_steps"]:.2f}')

# plot training reward + rolling average
plt.figure(figsize=(10, 6))
for noise in NOISE_LEVELS:
    rewards = training_rewards[noise]
    rolling = np.convolve(rewards, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='valid')
    plt.plot(np.arange(1, len(rolling) + 1), rolling, label=f'p={noise:.2f}')

plt.xlabel('Training Episode')
plt.ylabel(f'Rolling average reward ({WINDOW_SIZE} episodes)')
plt.title('Q-learning performance under different action noise levels')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/stochastic_learning_curves.png', dpi=300)
plt.show()

# plot evaluation success rates
plt.figure(figsize=(7, 5))
noise_labels = [f'{p:.2f}' for p in NOISE_LEVELS]
success_rates = [matched_results[p]['success_rate'] for p in NOISE_LEVELS]

plt.bar(noise_labels, success_rates)
plt.xlabel('Action noise probability')
plt.ylabel('Evaluation success rate')
plt.ylim(0, 1.05)
plt.title('Evaluation after training at each noise level')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/stochastic_evaluation.png', dpi=300)
plt.show()



# ------- Experiment 2: Test the p = 0 policy under noise -------
original_policy = trained_policies[0.0]
robustness_results = {}
print('\nTesting the original p=0 policy under added noise:')
for noise in NOISE_LEVELS:
    result = evaluate_policy(original_policy, noise, seed_offset=30000 + int(noise * 1000))
    robustness_results[noise] = result

    print(
        f'Test noise p={noise:.2f}: '
        f'success={result["success_rate"]:.1%}, '
        f'mean reward={result["mean_reward"]:.3f}, '
        f'average steps={result["mean_steps"]:.2f}'
    )


# compare original policy with policies trained at each noise level
plt.figure(figsize=(8, 5))
plt.plot(NOISE_LEVELS, [matched_results[p]['success_rate'] for p in NOISE_LEVELS], marker='o', label='Trained at matching noise level')
plt.plot(NOISE_LEVELS, [robustness_results[p]['success_rate'] for p in NOISE_LEVELS], marker='s', label='Original policy trained at p=0')
plt.xlabel('Evaluation action noise probability')
plt.ylabel('Success rate')
plt.xticks(NOISE_LEVELS)
plt.ylim(0, 1.05)
plt.title('Robustness of original policy to action noise')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/stochastic_robustness.png', dpi=300)
plt.show()
