import gymnasium as gym
import numpy as np
import os
import imageio.v2 as imageio

# best config based on hyperparameter tests
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.05
TRAIN_EPISODES = 2000

# eval settings
EVAL_EPISODES = 30
SEED = 42

# train Q-learning and return the learned Q-table
def train_q_learning():
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rng = np.random.default_rng(SEED)

    for episode in range(TRAIN_EPISODES):
        state, info = env.reset(seed=SEED + episode)
        done = False
        while not done:
            # epsilon-greedy action selection for train
            if rng.random() < EPSILON:
                action = rng.integers(n_actions)
            else:
                best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
                action = rng.choice(best_actions)

            next_state, reward, terminated, truncated, info = (env.step(action))

            if terminated:
                target = reward
            else:
                target = reward + GAMMA * np.max(Q[next_state])

            Q[state, action] += ALPHA * (target - Q[state, action])
            state = next_state
            done = terminated or truncated

    env.close()
    return Q


# evaluate the greedy policy without exploration
def evaluate_policy(Q):
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False, render_mode='rgb_array')
    rewards = []
    steps_per_episode = []
    video_frames = []

    for episode in range(EVAL_EPISODES):
        state, info = env.reset(seed=SEED + 10000 + episode)
        done = False
        total_reward = 0
        steps = 0

        # record the first evaluation episode
        if episode == 0:
            video_frames.append(env.render())

        while not done:
            # no exploration
            action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            if episode == 0:
                video_frames.append(env.render())

            total_reward += reward
            steps += 1
            state = next_state
            done = terminated or truncated

        rewards.append(total_reward)
        steps_per_episode.append(steps)

    env.close()

    # save recorded episode
    os.makedirs('results', exist_ok=True)
    video_path = 'results/frozenlake_agent.mp4'
    imageio.mimsave(video_path, video_frames, fps=3)
    print(f'Video saved to: {video_path}')

    return np.array(rewards), np.array(steps_per_episode)


# train agent
print('Training the Q-learning agent...')
Q = train_q_learning()

# evaluate learned policy
print('Evaluating the learned policy...')
rewards, steps = evaluate_policy(Q)
successes = int(np.sum(rewards))
average_reward = np.mean(rewards)
success_rate = average_reward * 100
print('\nPart G: Policy Evaluation Results')
print(f'Successful episodes: {successes}/{EVAL_EPISODES}')
print(f"Average reward: {average_reward:.2f}")
print(f'Success rate: {success_rate:.2f}%')
print(f'Average steps per episode: {np.mean(steps):.2f}')
