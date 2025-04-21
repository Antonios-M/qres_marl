import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .genetic_solver import GeneticAlgorithmSolver

# Assuming the genetic algorithm solver class (GeneticAlgorithmSolver) has been defined as before

def plot_rewards(rewards, completion_time, building_info, road_info):
    """Plot rewards over time and mark completion time, repair progress with circles for roads and squares for buildings."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(rewards)), y=[r for r in rewards], linestyle='-', color='b', label="Reward")

    # Mark the completion time if available
    if completion_time is not None:
        plt.axvline(x=completion_time, color='r', linestyle='--', label="Completed Repair Time")
        plt.text(completion_time, max([r for r in rewards]), 'Completed Repair', color='r', fontsize=12, verticalalignment='bottom')

    # Plot circles for roads and squares for buildings
    for road in road_info:
        plt.scatter(road[0], road[1], color='g', s=100, marker='o', label='Road')  # Example of road progress (x, reward)
    for building in building_info:
        plt.scatter(building[0], building[1], color='y', s=100, marker='s', label='Building')  # Example of building progress (x, reward)

    # Set x-axis ticks as integers and adjust step dynamically
    plt.xticks(np.arange(0, len(rewards), step=max(1, len(rewards) // 10), dtype=int))

    # Labels and title
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title("Reward Over Time Steps")
    plt.legend()

    # Show the plot
    plt.show()

# Example usage
def run_ga_ex(env):
    # Initialize the solver
    solver = GeneticAlgorithmSolver(env=env)

    # Run the genetic algorithm
    solver.run()

    # Get the best solution
    best_solution, best_fitness = solver.get_best_solution()
    print(f"Best solution: {best_solution}")
    print(f"Best fitness (total reward): {best_fitness}")

    # Run a rollout to collect rewards over time
    rewards = []
    state = solver.env.reset()
    completion_time = None  # You can calculate this based on the problem context
    building_info = []  # Add information specific to buildings
    road_info = []  # Add information specific to roads

    # Perform rollout using the best solution (policy)
    for t in range(500):  # Maximum of 500 timesteps
        action = best_solution
        action += 1
        next_state, reward, done, _, _ = solver.env.step(action)
        rewards.append(reward)
        state = next_state

        # Example: Track building and road information (for plotting purposes)
        building_info.append((t, reward))  # Dummy example
        road_info.append((t, reward))  # Dummy example

        if done:
            completion_time = t
            break

    # # Plot the rewards over time and mark the completion
    # plot_rewards(rewards, completion_time, building_info, road_info)
    print(f"Completion Time: {completion_time}")
    print(f"Building Info: {building_info}")
    print(f"Road Info: {road_info}")
    # Plot the rewards using the provided function
    print("Plotting rewards...")
    print(f"Rewards: {rewards}")
    solver.env.plot_rewards(
        rewards,
        completion_time,
        building_info,
        road_info,
        sum([solver.env.buildings_objs[i].max_income for i, _ in enumerate(solver.env.buildings_objs)]),
    )
