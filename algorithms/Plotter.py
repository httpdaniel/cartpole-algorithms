
import matplotlib
matplotlib.use("TkAgg")
from Genetic import CartPoleGenetic
from CartpoleDQN import CartpoleAgent
from PIDController import gradient_descent
from Baseline import main
import matplotlib.pyplot as plt
import gym

episodes = 110

# Baseline
baseline_values = main(episodes)

# Get values from PID algorithm
PID_values = gradient_descent(episodes=episodes)

#DQN
env = gym.make('CartPole-v1')
agent = CartpoleAgent(env)
DQN_values = agent.train(max_episodes=100000, episodes=episodes)

#Genetic
cart_pole_genetic = CartPoleGenetic(population_size=10, mutation_chance=0.1, mutation_value=1, render_result=False, weight_spread=2, mean_crossover = True)
cart_pole_genetic.train(episodes=episodes)
cart_pole_genetic.save_results()

# Plot
plt.plot([i for i in range(episodes)], baseline_values, label='Baseline Scores')
plt.plot([i for i in range(episodes)], PID_values, label='PID Scores')
plt.plot([i for i in range(episodes)], DQN_values, label='DQN Scores')
plt.plot([i for i in range(episodes)], cart_pole_genetic.best_scores, label='Genetic Scores')

plt.axhline(195, c='red', ls='--', label='goal')

plt.xlabel("Episodes")
plt.ylabel("Score by episode")
plt.legend()
plt.savefig('results/compound-plot')
plt.clf()
