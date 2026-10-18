import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
ALPHA = 0.1             # learning rate
GAMMA = 0.99            # discount factor
EPSILON = 0.3           # exploration rate
EPISODES = 2000
SEED = 42

# create environment
env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))

# Q-learning training loop
episode_rewards = []
rng = np.random.default_rng(SEED)
for episode in range(EPISODES):
    state, info = env.reset(seed=SEED+episode)
    done = False
    total_reward = 0

    while not done:
        # epsilon-greedy action selection
        if rng.random() < EPSILON:
            action = rng.integers(n_actions)
        else:
            # choose randomly among actions tied for the highest Q-value (to encourage new paths)
            best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
            action = rng.choice(best_actions)

        # take action in environment
        next_state, reward, terminated, truncated, info = (env.step(action))

        # Q-learning update
        if terminated:
            target = reward
        else:
            target = reward + GAMMA * np.max(Q[next_state])

        Q[state, action] += ALPHA * (target - Q[state, action])

        # track reward and move to next state
        total_reward += reward
        state = next_state

        # end episode if goal/hole/time limit
        done = terminated or truncated

    episode_rewards.append(total_reward)

env.close()


# calculate cumulative reward and rolling average reward across 100 episodes
episode_rewards = np.array(episode_rewards)
cumulative_rewards = np.cumsum(episode_rewards)
window_size = 100
rolling_average = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')

# plot cumulative reward
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax1.plot(np.arange(1, EPISODES + 1), cumulative_rewards, label='Cumulative reward')
ax1.set_title('FrozenLake Q-Learning - Training rewards')
ax1.set_ylabel('Cumulative reward')
ax1.legend()
ax1.grid(True, alpha=0.3)

# plot rolling average reward
ax2.plot(np.arange(window_size, EPISODES + 1), rolling_average, label='100 Episode rolling average')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Average reward')
ax2.set_ylim(-0.05, 1.05)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, EPISODES)
plt.tight_layout()
plt.show()

# print summary statistics
total_successes = int(cumulative_rewards[-1])
final_success_rate = np.mean(episode_rewards[-100:]) * 100
print(f'Total episodes: {EPISODES}')
print(f'Total successful episodes: {total_successes}')
print(f'Overall success rate: {np.mean(episode_rewards) * 100:.2f}%')
print(f'Success rate over final 100 episodes: {final_success_rate:.1f}%')
