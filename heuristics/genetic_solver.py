import gym
from gym.spaces.utils import flatten_space
import pygad
import numpy as np
from itertools import product

class GeneticAlgorithmSolver:
    def __init__(self,
        env: gym.Env,
        population_size=50,
        num_generations=100,
        crossover_probability=0.8,
        mutation_probability=0.2,
        mutation_type="swap"
    ) -> None:
        self.env = env.unwrapped
        self.num_genes = self.env.n_agents
        self.gene_space = np.array([a_i for a_i in range(61)])
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.mutation_type = mutation_type

        self.ga = None

    def fitness_func(self, pyga_instance, solution, solution_idx):
        """
        Fitness function to evaluate an individual's performance in the environment.

        Args:
            solution (np.array): The solution (chromosome), representing policy weights.
            solution_idx (int): The index of the solution in the population.

        Returns:
            float: The total reward accumulated during the simulation in the environment.
        """
        total_reward = 0
        state, info = self.env.reset()

        # Apply the policy (linear combination of state and chromosome weights)
        for _ in range(500):  # Maximum of 500 timesteps
            # # Ensure the solution is also flattened and compatible with the state
            # print(f"Solution: {solution}")
            # action = self.env.action_space.sample(solution)
            # Compute the action tuple
            # solution = solution[0]
            action = solution


            # action = solution
            # action = 1 if action > 0 else 0  # Choose action based on the sign of the result
            next_state, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            state = next_state

            # Flatten the next state to maintain consistency
            state = np.ravel(state)

            if done:
                break

        return total_reward

    def create_population(self, pop_size):
        """
        Creates the initial population of random solutions.

        Args:
            pop_size (int): The population size.

        Returns:
            np.array: The initial population (random weights for each individual).
        """
        return np.random.uniform(-1, 1, (pop_size, self.num_genes))

    def run(self):
        """
        Run the genetic algorithm to optimize the policy for the environment.
        """
        # Create the PyGAD GA object
        fitness_function = self.fitness_func
        self.ga = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.population_size // 2,
            fitness_func=fitness_function,
            sol_per_pop=self.population_size,
            num_genes=self.num_genes,
            gene_space=self.gene_space,
            gene_type=np.int32,
            parent_selection_type="tournament",
            keep_parents=5,
            crossover_type="uniform",
            crossover_probability=self.crossover_probability,
            mutation_type=self.mutation_type,
            mutation_probability=self.mutation_probability,
            initial_population=self.create_population(self.population_size)
        )

        # Run the genetic algorithm
        self.ga.run()

    def get_best_solution(self):
        """
        Get the best solution found after running the genetic algorithm.

        Returns:
            np.array: The best solution (policy weights).
            float: The fitness of the best solution.
        """
        best_solution, best_solution_fitness, _ = self.ga.best_solution()

        return best_solution, best_solution_fitness

    def test_best_solution(self):
        """
        Test the best solution found by the genetic algorithm in the environment.

        Returns:
            float: The total reward achieved by the best solution in the environment.
        """
        best_solution, _ = self.get_best_solution()
        total_reward = 0
        state, info = self.env.reset()

        for _ in range(500):  # Maximum of 500 timesteps
            action = best_solution
            next_state, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break

        return total_reward


# # Example usage
# if __name__ == "__main__":
#     # Instantiate the genetic algorithm solver
#     env = gym.make("CartPole-v1")  # Replace with your environment
#     solver = GeneticAlgorithmSolver(env)

#     # Run the genetic algorithm
#     solver.run()

#     # Get and print the best solution
#     best_solution, best_fitness = solver.get_best_solution()
#     print(f"Best solution: {best_solution}")
#     print(f"Best fitness (total reward): {best_fitness}")

#     # Test the best solution in the environment
#     total_reward = solver.test_best_solution()
#     print(f"Total reward achieved with the best solution: {total_reward}")
