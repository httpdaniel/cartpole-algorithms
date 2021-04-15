import gym
import numpy as np
import random
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import os

class CartPoleGenetic:
    def __init__(self, population_size=10, weight_spread=2, crossover_individuals=4,
                    mutation_chance=0.05, mutation_value=1,
                        random_seed=0, elitism=0.2, convergence_condition=195, render_result=True, mean_crossover=False):
        self.env = gym.make("CartPole-v1")
        self.env.seed(random_seed)
        random.seed(random_seed)

        self.population_size = population_size
        self.weight_spread = weight_spread
        self.render_result = render_result
        self.crossover_individuals = crossover_individuals
        self.mutation_chance = mutation_chance
        self.mutation_value = mutation_value
        self.elitism = elitism
        self.convergence_condition = convergence_condition
        self.convergence_count = 0
        self.mean_crossover = mean_crossover
        self.scores = []
        self.best_scores = []
        self.solved = False
        self.init_population()

    def init_population(self):
        self.population = []

        for _ in range(self.population_size):
            random_weights = [random.uniform(-self.weight_spread/2, self.weight_spread/2) for _ in range(4)]
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

    def select_parent(self):
        candidates = []
        while len(candidates) < self.crossover_individuals:
            candidate = random.randrange(self.population_size)
            if(candidate not in candidates):
                candidates.append(candidate)
        # I return the candidate with the highest score 
        # the population is already sorted by score, so I just return the element with the smallest index
        return min(candidates)
    
    def train(self):
        iteration_count = 0
        start = time()

        while not self.solved:
            iteration_count += 1
            self.create_next_generation()

        end = time()
        self.time_taken = end - start

        self.episodes = iteration_count-100
        print('iteration count:', self.episodes)
        print('time taken', self.time_taken)

    def save_results(self):
        # Save results to file
        if not os.path.exists('results'):
            os.makedirs('results')

        best_individual = np.max(self.train_generation())
        df = pd.DataFrame(data=[[best_individual, self.episodes, np.mean(self.scores[-100:]), self.time_taken/60]], index=None, 
                                            columns=['Final Reward', 'Number Of Episodes', 'Average Reward', 'Time Taken'])
        df.to_csv('results/ga-results.csv', index=False)

    def plot_results(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=200)
        fig.suptitle("Genetic Algorithm Results")
        ax[0].plot(self.scores, label='score per run')
        ax[0].axhline(195, c='red', ls='--', label='goal')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Reward')
        x = range(len(self.scores))
        ax[0].legend()

        # Calculate the trend
        try:
            z = np.polyfit(x, self.scores, 1)
            p = np.poly1d(z)
            ax[0].plot(x, p(x), "--", label='trend')
        except:
            print('')

        # Plot the histogram of results
        ax[1].hist(self.scores[-50:])
        ax[1].axvline(195, c='red', label='goal')
        ax[1].set_xlabel('Scores for last 50 Episodes')
        ax[1].set_ylabel('Frequency')
        ax[1].legend()
        #plt.show()

    def generate_crossover_individual(self):
        parent_1_index = self.select_parent()
        while True:
            parent_2_index = self.select_parent()
            if parent_2_index != parent_1_index:
                break

        parent_1 = self.population[parent_1_index]
        parent_2 = self.population[parent_2_index]

        if(self.mean_crossover):
            crossover_individual_1 = [(parent_1[0] + parent_2[0])/2, (parent_1[1] + parent_2[1])/2,
                                        (parent_1[2] + parent_2[2])/2, (parent_1[3] + parent_2[3])/2]
            return [crossover_individual_1]
        else:
            # I take the first 2 weights from one parent and the others from the other parent
            crossover_individual_1 = [parent_1[0], parent_1[1], parent_2[2], parent_2[3]]
            crossover_individual_2 = [parent_2[0], parent_2[1], parent_1[2], parent_1[3]]
            return [crossover_individual_1, crossover_individual_2]

    def mutate_generation(self):
        for weights in self.population:
            for i in range(4):
                if random.uniform(0, 1) < self.mutation_chance:
                    weights[i] += random.uniform(-self.mutation_value/2, self.mutation_value/2)

    def create_next_generation(self):
        fitness_score_list = self.train_generation()

        # sorting values
        self.population = [weights for _, weights in sorted(zip(fitness_score_list, self.population))]
        self.population = list(reversed(self.population)) # from highest to lowest fitness_value
        fitness_score_list.sort(reverse=True) # I also sort the fitness_scores, so they match the order of the population
        
        if(np.mean(fitness_score_list) >= self.convergence_condition):
            self.convergence_count += 1
        else:
            self.convergence_count = 0

        self.mean_score = np.mean(fitness_score_list)
        self.scores.append(self.mean_score)
        self.best_scores.append(np.max(fitness_score_list))
        print(self.mean_score, self.best_scores[-1], np.mean(self.scores[-100:]))

        top_performers_number = int(self.elitism * self.population_size)
        top_performers = [self.population[i] for i in range(top_performers_number)]

        next_generation = []
        while(len(next_generation) < self.population_size-top_performers_number):
            individuals = self.generate_crossover_individual()
            next_generation.append(individuals[0])
            if(len(next_generation) < self.population_size and len(individuals) == 2):
                next_generation.append(individuals[1])
        self.population = next_generation
        self.mutate_generation()

        for top_performer in top_performers:
            self.population.append(top_performer)

        if(len(self.scores) >= 100):
            if np.mean(self.scores[-100:]) >= self.convergence_condition:
                self.solved = True

def main():      
    cart_pole_genetic = CartPoleGenetic(population_size=10, mutation_chance=0.1, mutation_value=1, render_result=False, weight_spread=2, mean_crossover = True)
    cart_pole_genetic.train()
    cart_pole_genetic.save_results()
    cart_pole_genetic.plot_results()
    plt.savefig('results/ga-plot')
    plt.clf()

    # Save Chart to file
    plt.plot([i for i in range(len(cart_pole_genetic.scores))], cart_pole_genetic.scores, label='Scores')
    plt.xlabel("Episodes")
    plt.ylabel("Mean Population Score by episode")
    plt.legend()
    plt.savefig('results/ga-plot-mean')
    plt.clf()

    # Save Chart to file
    plt.plot([i for i in range(len(cart_pole_genetic.best_scores))], cart_pole_genetic.best_scores, label='Scores')
    plt.xlabel("Episodes")
    plt.ylabel("Mean Population Score by episode")
    plt.legend()
    plt.savefig('results/ga-plot-best')
    plt.clf()

if __name__ == "__main__":
    main()
