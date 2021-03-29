import gym
import numpy as np
import random
from time import sleep

class CartPoleGenetic:
    def __init__(self, population_size=10, weight_spread=2, random_seed=0, render_result=True):
        self.env = gym.make("CartPole-v1")
        self.env.seed(random_seed)
        random.seed(random_seed)

        self.population_size = population_size
        self.weight_spread = weight_spread
        self.render_result = render_result
        self.init_population()

    def init_population(self):
        self.population = []

        for _ in range(self.population_size):
            random_weights = [random.uniform(0, self.weight_spread) for _ in range(4)]
            self.population.append(random_weights)

    @staticmethod
    def select_action(weights, observation):
        result = 0
        for i in range(4):
            result += weights[i] * observation[i]
        
        return 0 if result < 0.5 else 1

    def fitness(self, weights):
        observation = self.env.reset()
        fitness_score = 0
        done = False

        while not done:
            if self.render_result:
                self.env.render()
            sleep(0.005)
            action = CartPoleGenetic.select_action(weights, observation)
            observation, reward, done, _ = self.env.step(action)
            fitness_score += reward

        return fitness_score

    def train_generation(self):
        fitness_score_list = []

        for weights in self.population:
            fitness_score = self.fitness(weights)
            fitness_score_list.append(fitness_score)
        return fitness_score_list

    def crossover_generation(self):
        pass

    def mutate_generation(self):
        pass 

    def create_next_generation(self):
        fitness_score_list = self.train_generation()

        # sorting values
        self.population = [weights for _, weights in sorted(zip(fitness_score_list, self.population))]
        self.population = list(reversed(self.population)) # from highest to lowest fitness_value
        print(fitness_score_list)








cart_pole_genetic = CartPoleGenetic(weight_spread=1)
for _ in range(1):
    cart_pole_genetic.create_next_generation()
