# Import packages
import gym
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tqdm import tqdm

# Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.005

BATCH_SIZE = 32
MEMORY_SIZE = 100000

EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995


class DQN:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.exploration_rate = EPSILON
        self.model = self.create_model()

    # Create keras model
    def create_model(self):
        model = Sequential()
        model.add(Input((self.state_space,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(LEARNING_RATE))
        return model

    # Get next action to take
    def update_action(self, state):
        state = np.reshape(state, [1, self.state_space])
        self.exploration_rate *= EPSILON_DECAY
        self.exploration_rate = max(self.exploration_rate, EPSILON_MIN)
        if np.random.random() < self.exploration_rate:
            return random.randint(0, self.action_space - 1)
        q_value = self.predict(state)[0]
        return np.argmax(q_value)

    # Fit the model
    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)

    # Predict state
    def predict(self, state):
        return self.model.predict(state)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # Save details of last step
    def save(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    # Randomly sample from buffer
    def sample(self):
        sample = random.sample(self.buffer, BATCH_SIZE)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states)
        states = states.reshape(BATCH_SIZE, -1)
        next_states = np.array(next_states)
        next_states = next_states.reshape(BATCH_SIZE, -1)
        return states, actions, rewards, next_states, done

    # Return size of buffer
    def size(self):
        return len(self.buffer)


class CartpoleAgent:
    def __init__(self, env):
        self.env = env
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

        self.model = DQN(self.state_space, self.action_space)
        self.target = DQN(self.state_space, self.action_space)
        self.update_weights()

        self.buffer = ReplayBuffer(capacity=10000)

    def update_weights(self):
        weights = self.model.model.get_weights()
        self.target.model.set_weights(weights)

    # Replay from buffer
    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target.predict(states)
            next_values = self.target.predict(next_states).max(axis=1)
            targets[range(BATCH_SIZE), actions] = rewards + (1 - done) * next_values * GAMMA
            self.model.train(states, targets)

    # Train the model for x iterations
    def train(self, max_episodes):
        start_time = time.time()
        reward_count = 0
        final = []
        for ep in tqdm(range(max_episodes)):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                if ep % 10 == 0:
                    self.env.render()
                action = self.model.update_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.save(state, action, reward * 0.01, next_state, done)
                total_reward += reward
                state = next_state
            self.env.close()
            if self.buffer.size() >= BATCH_SIZE:
                self.replay()
            self.update_weights()
            print('Episode{} Reward={} Count={}'.format(ep, total_reward, reward_count))
            final.append(total_reward)

            # Check for convergence - reward greater than 195 for 100 iterations in a row
            if total_reward >= 195:
                reward_count = reward_count + 1
            else:
                reward_count = 0
            if reward_count >= 100:
                end_time = time.time()
                time_taken = end_time - start_time
                av_reward = sum(final)/(ep+1)
                plot_results(final)
                get_results(total_reward, ep+1, av_reward, time_taken)
                break


def main():
    env = gym.make('CartPole-v1')
    agent = CartpoleAgent(env)
    agent.train(max_episodes=1000)


# Plot results for each iteration
def plot_results(values):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=200)
    fig.suptitle("DQN Results")
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()

    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores for last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()


# Export results to csv
def get_results(total_reward, ep, av_reward, time_taken):
    minutes = time_taken/60
    minutes = '%.2f' % minutes

    res = {'Final Reward': total_reward,
           'Number of episodes': ep,
           'Average Reward': av_reward,
           'Time Taken (Minutes)': minutes
           }

    res_df = pd.DataFrame([res], columns=['Final Reward', 'Number of episodes', 'Average Reward', 'Time Taken (Minutes)'
                                          ])
    res_df.to_csv('../results/dqn-results.csv', index=False, encoding='utf-8')


if __name__ == "__main__":
    main()
