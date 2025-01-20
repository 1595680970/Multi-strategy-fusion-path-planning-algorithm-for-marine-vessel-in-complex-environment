# Multi-strategy Fusion Path Planning Algorithm for Autonomous Surface Vessels in Complex Environments

This project proposes a path planning algorithm that integrates an improved Fast Marching Method (FMM) with the Dynamic Window Approach (DWA) for Autonomous Surface Vessels (ASVs), designed to navigate dynamic obstacles in complex environments.

This project proposes a path planning algorithm that integrates an improved Fast Marching Method (FMM) with the Dynamic Window Approach (DWA) for Autonomous Surface Vessels (ASVs), designed to navigate dynamic obstacles in complex environments.

# Path Planning Algorithms Repository

This repository contains code implementations and simulation results for path planning algorithms designed for Autonomous Surface Vessels (ASVs) in dynamic environments. The project enhances the traditional Fast Marching Method (FMM) by integrating it with the Dynamic Window Approach (DWA), as well as comparing it with traditional path planning algorithms such as A*, Dijkstra, and Rapidly-exploring Random Trees (RRT).

## Project Structure

### Enhancement of Fusion Algorithm in Dynamic Environments
The **"Enhancement of Fusion Algorithm in Dynamic Environments"** folder contains the following files:

1. **Experimental Setup Diagram1**: Code for the first experimental scenario, illustrating the setup for initial path planning simulations.
2. **Experimental Setup Diagram2**: Code for the second experimental scenario, featuring a different configuration for testing path planning performance.
3. **Improved FMM Path-Planning Performance**: Code to demonstrate the performance of the enhanced FMM algorithm in dynamic environments, showing improvements in efficiency over traditional methods.
4. **Fusion Algorithm + Random-position Obstacles**: Code implementing the fusion algorithm with a simplified vessel model, where obstacles are randomly placed in the environment. This simulation evaluates how the hybrid algorithm handles dynamic obstacles.
5. **Fusion Algorithm + Trajectories for Visualization and Analysis**: Code for the fusion algorithm with fixed-position obstacles. The simulation tracks and visualizes the vessel's trajectory, outputting several key metrics including:
   - ASV's velocity over time
   - Control inputs over time
   - Distance to target over time
   - Velocity vs position coordinate
   - Control inputs vs position coordinate
6. **single-DWA**: Code for simulating the Dynamic Window Approach (DWA) algorithm alone, illustrating its path planning results and limitations.
7. **single-FMM**: Code for simulating the Fast Marching Method (FMM) algorithm alone, showcasing its performance as a standalone path planning solution.

### The Compares Path Planning Algorithm
The **"The Compares Path Planning Algorithm"** folder contains the following files:

1. **A-star**: Code for implementing the A* algorithm, a widely used pathfinding and graph traversal algorithm that finds the shortest path between two points.
2. **Dijkstra**: Code for implementing the Dijkstra algorithm, another shortest path algorithm that calculates the minimum distance between nodes in a graph.
3. **FMM**: Code for simulating the Fast Marching Method (FMM) algorithm, used for efficient pathfinding by solving the Eikonal equation of wave propagation.
4. **RRT**: Code for implementing the Rapidly-exploring Random Tree (RRT) algorithm, a popular method for path planning in high-dimensional spaces, commonly used for motion planning.

## Key Improvements
This project introduces the following key improvements:

1. **Enhanced Fusion Algorithm**: By combining the FMM and DWA algorithms, the fusion approach optimizes path planning in dynamic environments, providing a more robust solution compared to traditional methods.
2. **Comprehensive Visualization**: The integration of detailed trajectory visualizations allows for deeper analysis of the vessel's behavior during the path planning process.
3. **Better Dynamic Obstacle Handling**: The hybrid algorithm demonstrates improved performance in handling dynamic obstacles, with both random and fixed-position obstacle scenarios included in the simulations.
4. **Clear Documentation**: The repository contains clear and concise documentation for each component, making it easier for users to understand and utilize the code for their own simulations.
5. **Consistent Structure and Naming**: Consistent file naming and code structure enhance the overall readability and navigability of the repository.

## License

This project is licensed under the Apache License 2.0, with the addition of the Commons Clause License. Please refer to the LICENSE file for more details.

## License Acknowledgements

This software includes libraries licensed under the Apache License 2.0 and the Commons Clause License. For more details, please see the LICENSE file.

## Requirements

To run the code in this repository, you will need to install the following Python libraries:

```bash
pip install pygame numpy matplotlib scipy cartopy shapely
