# Cartpole Algorithms
This repository contains a collection of algorithms used to solve the Cartpole environment on [OpenAI Gym](https://gym.openai.com/envs/CartPole-v1/).

## Cartpole

<p align="center">
  <img src="https://github.com/httpdaniel/cartpole-algorithms/blob/main/assets/cartpole-initial.png" width="300">
</p>

Cartpole balancing is a control problem where the goal is to balance a pole upright on the top of a cart by applying the appropriate forces left and right.

States:
* The position of the cart
* The velocity of the cart
* The angle of the pole
* The angular velocity of the pole

Actions:
* Apply a force of +1 (move the cart right)
* Apply a force of -1 (move the cart left)

## The Algorithms
Three algorithms were used for this project:
* A Deep Q-Network with Experience Replay
* A Genetic Algorithm
* A PID Controller with Gradient Descent

## Project Dependencies
1. Python3 (Note: ensure you have python mapped to the python3 environment variable)
2. NumPy
3. Scikit-learn (Note: ensure 0.22.1 is the version of Scikit-learn being used)
4. Matplotlib (Note: ensure that your operating system is able to use the TKAgg function)
5. Keras (Note: ensure you are using a TensorFlow backend and 2.4.3 is the version of Keras used)
6. TensorFlow (Note: ensure 2.4.1 is the version of TensorFlow being used)
7. Pillow
8. pip (Note: ensure you have pip mapped to the pip3 environment variable)

## Install The Dependencies
To install the project dependencies run the command "pip3 install -r requirements.txt"

## Running The Project
The following steps detail how to run the project:
1. Use the cd command in order to navigate to the "algorithms" directory
2. Run python Plotter.py to see the results of all the algorithms and Baseline
3. Run python Baseline.py to see the results of the Baseline alone
4. Run python CartpoleDQN.py to see the results of the DQN alone
5. Run python Genetic.py to see the results of the Genetic Algorithm alone
6. Run python PIDController.py to see the results of the PID Controller alone

## Project
MSc Computer Science - Intelligent Systems
Module:  CS7IS2 - Artificial Intelligence

### Students

Name: Claire Farrell
Student Number: 1619148

Name: Daniel Farrell
Student Number: 18315021

Name: Joshua Cassidy
Student Number: 20300057

Name: Matteo Bresciani
Student Number: 20309566
