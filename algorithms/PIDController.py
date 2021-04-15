import gym
import numpy as np
import random
import time
random.seed(7)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_descent(learning_rate = 0.05, epochs = 200, episodes=0):
    best_pid = None
    total_reward = 0
    av_reward = 0
    episode = 0
    time_taken = 0
    final = 0
    prev_reward = None
    start_time = time.time()

    for _ in range(10):
        P, I, D = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
        history = []
        for i in range(epochs):
            
            reward = pid(P, I, D)
            p_temp = P - (learning_rate * reward["error"][0])
            i_temp = I - (learning_rate * reward["error"][1])
            d_temp = D - (learning_rate * reward["error"][2])

            if p_temp > 0 and p_temp < 1:
                P = P - (learning_rate * reward["error"][0])
            
            if i_temp > 0 and i_temp < 1:
                I = I - (learning_rate * reward["error"][1])

            if d_temp > 0 and d_temp < 1:
                D = D - (learning_rate * reward["error"][2])

            if not best_pid or best_pid["reward"] < reward["total_reward"]:
                
                best_pid = {
                    "P": P,
                    "I": I,
                    "D": D,
                    "reward": reward["total_reward"]
                }
                total_reward = reward["total_reward"]
                av_reward = reward["av_reward"]
                episode = reward["episode"]
                final = reward["final"]
                
            prev_reward = reward
            if reward["total_reward"] >=  500:
                break
            
            if len(history) > 100 and np.array_equal(np.gradient(history[len(history)-100:]), [0 for i in range(100)]):
                break
            history.append(reward["total_reward"])
    end_time = time.time()
    time_taken = end_time - start_time
    plot_results(final)
    get_results(total_reward, episode, av_reward, time_taken)
    return final
          
def get_results(total_reward, ep, av_reward, time_taken):
    minutes = time_taken/60
    minutes = '%.2f' % minutes

    res = {
        'Final Reward': total_reward,
        'Number of episodes': ep,
        'Average Reward': av_reward,
        'Time Taken (Minutes)': minutes
    }

    res_df = pd.DataFrame([res], columns=['Final Reward', 'Number of episodes', 'Average Reward', 'Time Taken (Minutes)'
                                          ])
    res_df.to_csv('results/pid-results.csv', index=False, encoding='utf-8')


def plot_results(values):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=200)
    fig.suptitle("PID Results")
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()

    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')
    episodes_count = len(values[-50:])
    ax[1].hist(values[-50:], width=10 )
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores for last %s Episodes' % episodes_count)
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.savefig("results/pid-results-plot.png")
    plt.show()


def pid(P, I, D, episodes=200): 
    cartpole_env = gym.make('CartPole-v1')
    cartpole_env._max_episode_steps = 500
    cartpole_env.seed(7)
    current_state = cartpole_env.reset()
    done = False
    total_reward = 0
    final = []
    reward_count = 0

    goal_state = np.array([0, 0, 0, 0])
    for episode in range(episodes):
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
        
        final.append(total_reward)
        
        if len(final) > 100:
            if np.mean(final[-100:]) >= 195:
                av_reward = sum(final)/(episode+1)
                cartpole_env.close()
                return {
                    "total_reward": total_reward,
                    "error": error,
                    "episode": episode+1,
                    "av_reward": av_reward,
                    "final": final,
                }

    av_reward = sum(final)/(episode+1)
    cartpole_env.close()
    return {
        "total_reward": total_reward,
        "error": error,
        "episode": episode+1,
        "av_reward": av_reward,
        "final": final
    }

def main():
    gradient_descent()

if __name__ == "__main__":
    main()

