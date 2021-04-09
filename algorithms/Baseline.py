# Import packages
import gym
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
random.seed(7)

class CartpoleAgent:
    def __init__(self, env):
        self.env = env
        self.env.seed(7)
        self.action_space = self.env.action_space.n

    # Select random action from action space
    def update_action(self):
        action = random.randint(0, self.action_space - 1)
        return action


def main():
    env = gym.make('CartPole-v1')
    agent = CartpoleAgent(env)
    episodes = 200

    final = []
    start_time = time.time()
    for ep in tqdm(range(episodes)):
        state = env.reset()
        done, total_reward = False, 0
        while not done:
            if ep % 10 == 0:
                env.render()
            action = agent.update_action()
            state, reward, done, _ = env.step(action)
            total_reward += reward
        env.close()
        print('Episode{} Reward={}'.format(ep, total_reward))
        final.append(total_reward)
        if ep == episodes-1:
            end_time = time.time()
            time_taken = end_time - start_time
            av_reward = sum(final)/(ep+1)
            plot_results(final)
            get_results(total_reward, ep+1, av_reward, time_taken)
            break


# Plot results for each iteration
def plot_results(values):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=200)
    fig.suptitle("Random search baseline")
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
    res_df.to_csv('../results/baseline-results.csv', index=False, encoding='utf-8')


if __name__ == "__main__":
    main()
