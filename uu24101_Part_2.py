# importing necessary libraries
import numpy as np
import matplotlib as plt

class Agent:
    # infectious agent in the SEIR monte carlo model,
    # this stores the position and its compartment

    # using numbers to represent each state
    EMPTY = 0
    SUSCEPTIBLE = 1
    EXPOSED = 2 
    INFECTED = 3
    RECOVERED = 4

    # method to initialise and create a new Agent.
    def __init__(self, position, state):
        self.xpos = position[0]
        self.ypos = position[1]
        self.state = state

    
    



