import gym
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

cartpole_env = gym.make('CartPole-v1')
cartpole_env._max_episode_steps = 10000
cartpole_env.seed(7)
current_state = cartpole_env.reset()
done = False
total_reward = 0
while not done:
    action = np.round(sigmoid(random.uniform(-1,1)))
    current_state, reward, done, debug_info = cartpole_env.step(action.astype(np.int32))
    total_reward += reward
print(total_reward)
cartpole_env.close()
