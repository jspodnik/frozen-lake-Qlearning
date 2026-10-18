import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# fixed hyperparameters
ALPHA = 0.1             
GAMMA = 0.99
#EPSILON = 0.3
EPISODES = 2000
SEED = 42
WINDOW_SIZE = 100

# hyperparameter to test
EPSILON_VALUES = [0.05, 0.2, 0.5]

# train Q-learning using the specified hyperparameter value
def train_q_learning(epsilon):
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    episode_rewards = []

    # use the same random seed for each experiment
    rng = np.random.default_rng(SEED)

    for episode in range(EPISODES):
        state, info = env.reset(seed=SEED+episode)
        done = False
        total_reward = 0
        while not done:
            # epsilon-greedy action selection
            if rng.random() < epsilon:
                action = rng.integers(n_actions)
            else:
                # choose randomly among actions tied for the highest Q-value (to encourage new paths)
                best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
                action = rng.choice(best_actions)

            # execute action
            next_state, reward, terminated, truncated, info = (env.step(action))

            # Q-learning update
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
    return np.array(episode_rewards)


# train and record results for each value of the varying hyperparameter
results = {}
for epsilon in EPSILON_VALUES:
    print(f'Training with epsilon = {epsilon}...')
    results[epsilon] = train_q_learning(epsilon)

# plot rolling average reward for all configurations
plt.figure(figsize=(12, 6))
for epsilon, rewards in results.items():
    rolling_average = np.convolve(rewards, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='valid')
    episodes = np.arange(WINDOW_SIZE, EPISODES + 1)
    plt.plot(episodes, rolling_average, label=f'Epsilon = {epsilon}')

plt.title('Rolling average reward for different discount factors')
plt.xlabel('Episode')
plt.ylabel('Average reward per episode')
plt.xlim(WINDOW_SIZE, EPISODES)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
