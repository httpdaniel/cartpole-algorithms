import gym
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

P = 1
I = 0.1
D = 0.5
cartpole_env = gym.make('CartPole-v1')
cartpole_env._max_episode_steps = 10000
cartpole_env.seed(7)
current_state = cartpole_env.reset()
done = False
total_reward = 0
goal_state = np.array([0, 0, 0, 0])
for episode in range(20):
    total_reward = 0
    current_state = cartpole_env.reset()
    integral = 0
    derivative = 0
    previous_error = 0
    done = False
    while not done:
        error = current_state - goal_state
        integral += error
        derivative = error - previous_error

        pid = np.dot(np.dot(P, error) + np.dot(I, integral) + np.dot(D, derivative), np.array([0, 0, 1, 0]))
        action = np.round(sigmoid(pid))
        
        current_state, reward, done, debug_info = cartpole_env.step(action.astype(np.int32))
        total_reward += reward
        previous_error = error

print(total_reward)
cartpole_env.close()
