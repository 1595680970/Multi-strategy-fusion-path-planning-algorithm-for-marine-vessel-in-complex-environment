# Multi-strategy fusion path planning algorithm for autonomous surface vessels in complex environment
This project proposes a path planning algorithm that integrates an improved Fast Marching Method (FMM) with the Dynamic Window Approach (DWA) for ASV. 

# Path Planning Algorithms Repository

This repository contains code implementations and simulation results for path planning algorithms used in dynamic environments for Autonomous Surface Vessels (ASVs). The project includes improvements on the traditional Fast Marching Method (FMM) by integrating it with the Dynamic Window Approach (DWA), as well as comparisons with traditional path planning algorithms such as A*, Dijkstra, and Rapidly-exploring Random Trees (RRT).

## Project Structure

### Enhancement of Fusion Algorithm in Dynamic Environments
The "Enhancement of Fusion Algorithm in Dynamic Environments" folder contains the following files:

1. **Experimental Setup Diagram1**: Code for the first experimental scenario, illustrating the initial setup for path planning simulations.

2. **Experimental Setup Diagram2**: Code for the second experimental scenario, showing a different configuration for testing path planning performance.

3. **Improved FMM Path-Planning Performance**: Code to demonstrate the performance of the improved FMM algorithm in dynamic environments, showcasing its efficiency and optimization over traditional methods.

4. **Fusion Algorithm + Random-position Obstacles**: Code that implements the fusion algorithm with a simplified vessel model, where obstacles are randomly placed in the environment. This simulation evaluates how the hybrid algorithm handles dynamic obstacles.

5. **Fusion Algorithm + Trajectories for Visualization and Analysis**: Code for the fusion algorithm with fixed-position obstacles, where the simulation tracks and visualizes the vessel's trajectory. The program outputs several key metrics, including:
   - ASV's velocity over time
   - Control inputs over time
   - Distance to target over time
   - Velocity vs position coordinate
   - Control inputs vs position coordinate

6. **single-DWA**: Code for simulating the Dynamic Window Approach (DWA) algorithm by itself, highlighting its path planning results and limitations.

7. **single-FMM**: Code for simulating the Fast Marching Method (FMM) algorithm on its own, presenting its performance as a standalone path planning solution.

### The Compares Path Planning Algorithm
The "The Compares Path Planning Algorithm" folder contains the following files:

1. **A-star**: Code for implementing the A* algorithm, a widely used pathfinding and graph traversal algorithm that finds the shortest path between two points.

2. **Dijkstra**: Code for implementing the Dijkstra algorithm, another shortest path algorithm that calculates the minimum distance between nodes in a graph.

3. **FMM**: Code for simulating the Fast Marching Method (FMM) algorithm, which is used for efficient pathfinding by solving the Eikonal equation of wave propagation.

4. **RRT**: Code for implementing the Rapidly-exploring Random Tree (RRT) algorithm, a popular method for path planning in high-dimensional spaces, typically used for motion planning.

## Requirements

To run the code in this repository, the following Python libraries are required:

```bash
pip install pygame numpy matplotlib scipy cartopy shapely
