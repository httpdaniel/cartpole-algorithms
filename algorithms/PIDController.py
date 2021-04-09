import gym
import numpy as np
import random
import time
random.seed(7)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_descent(learning_rate = 0.05, epochs = 1000):
    best_pid = None
    total_reward = 0
    av_reward = 0
    episode = 0
    time_taken = 0
    final = 0
    prev_reward = None
   
    P, I, D = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
    for i in range(epochs):
        start_time = time.time()
        reward = pid(P, I, D)

        P = P - (learning_rate * reward["error"][0])
        I = I - (learning_rate * reward["error"][1])
        D = D - (learning_rate * reward["error"][2])
        end_time = time.time()
        
        if not best_pid or best_pid["reward"] < reward["total_reward"]:
            time_taken = end_time - start_time
            best_pid = {
                "P": P,
                "I": I,
                "D": D,
                "reward": reward["total_reward"],
                "time_taken": time_taken
            }
            total_reward = reward["total_reward"]
            av_reward = reward["av_reward"]
            episode = reward["episode"]
            final = reward["final"]
        prev_reward = reward
        if reward["total_reward"] >=  1000:
            break
    print(best_pid)        
          


def pid(P, I, D): 
    cartpole_env = gym.make('CartPole-v1')
    cartpole_env._max_episode_steps = 10000
    cartpole_env.seed(7)
    current_state = cartpole_env.reset()
    done = False
    total_reward = 0
    final = []
    reward_count = 0

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
        
        final.append(total_reward)
        
        
        if total_reward >= 195:
            reward_count = reward_count + 1
        else:
            reward_count = 0
        if reward_count >= 100:
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
    

gradient_descent()
