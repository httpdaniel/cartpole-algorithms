from Genetic import CartPoleGenetic
from CartpoleDQN import CartpoleAgent
from PIDController import gradient_descent
import matplotlib.pyplot as plt
import gym

# Get values from PID algorithm
PID_values = gradient_descent()

#DQN
env = gym.make('CartPole-v1')
agent = CartpoleAgent(env)
DQN_values = agent.train(max_episodes=100000)

#Genetic
cart_pole_genetic = CartPoleGenetic(population_size=10, mutation_chance=0.1, mutation_value=1, render_result=False, weight_spread=2, mean_crossover = True)
cart_pole_genetic.train()
cart_pole_genetic.save_results()

# Plot
plt.plot([i for i in range(len(PID_values))], PID_values, label='PID Scores')
plt.plot([i for i in range(len(DQN_values))], DQN_values, label='DQN Scores')
plt.plot([i for i in range(len(cart_pole_genetic.scores))], cart_pole_genetic.scores, label='Genetic Scores')

plt.axhline(195, c='red', ls='--', label='goal')

plt.xlabel("Episodes")
plt.ylabel("Score by episode")
plt.legend()
plt.savefig('results/compound-plot')
plt.clf()
