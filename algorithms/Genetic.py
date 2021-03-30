import gym
import numpy as np
import random

class CartPoleGenetic:
    def __init__(self, population_size=10, weight_spread=2, crossover_individuals=4,
                    mutation_chance=0.05, mutation_value=1,
                        random_seed=0, elitism=0.2, render_result=True):
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

    def generate_crossover_individual(self):
        parent_1_index = self.select_parent()
        while True:
            parent_2_index = self.select_parent()
            if parent_2_index != parent_1_index:
                break

        parent_1 = self.population[parent_1_index]
        parent_2 = self.population[parent_2_index]
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

        print(np.mean(fitness_score_list))

        top_performers_number = int(self.elitism * self.population_size)
        top_performers = [self.population[i] for i in range(top_performers_number)]

        next_generation = []
        while(len(next_generation) < self.population_size-top_performers_number):
            individuals = self.generate_crossover_individual()
            next_generation.append(individuals[0])
            if(len(next_generation) < self.population_size):
                next_generation.append(individuals[1])
        self.population = next_generation

        for top_performer in top_performers:
            self.population.append(top_performer)
        
        self.mutate_generation()



cart_pole_genetic = CartPoleGenetic(population_size=20, render_result=False)
for _ in range(100):
    cart_pole_genetic.create_next_generation()
