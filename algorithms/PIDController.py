import gym

cartpole_env = gym.make('CartPole-v1')
cartpole_env._max_episode_steps = 10000
cartpole_env.seed(7)
current_state = cartpole_env.reset()
action = 1
done = False
total_reward = 0
while not done:
    current_state, reward, done, debug_info = cartpole_env.step(action)
    total_reward += reward
print(total_reward)
cartpole_env.close()
