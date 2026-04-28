# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

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

    # method to allow agents to move to neighbouring lattice cell.
    # if occupied, agent remains at original cell. 
    def move(self, lattice, rng):
        numlattice = lattice.shape[0]

        # moving the agent in a random direction to a neighbouring cell
        directions = [(1,0), (0,1), (-1,0), (0, -1)]
        dx, dy = directions[rng.integers(0, 4)]

        # updating posiiton of the agent
        newx = self.xpos + dx
        newy = self.ypos + dy
        
        # since I'm using a periodic boundary 
        # I've used the modulus operator to handle edge cases
        newx = newx % numlattice
        newy = newy % numlattice

        if lattice[newx, newy] == Agent.EMPTY:
            lattice[self.xpos, self.ypos] = Agent.EMPTY

            self.xpos = newx
            self.ypos = newy

            lattice[self.xpos, self.ypos] = self.state

    def infected_neighbour(self, lattice):
        
        # method to check if neighbours are infected agents. 

        numlattice = lattice.shape[0]

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in directions:
            nx = self.xpos + dx
            ny = self.ypos + dy

            nx = nx % numlattice
            ny = ny % numlattice

            if lattice[nx, ny] == Agent.INFECTED:
                return True

        return False
    
    def update_state(self, lattice, rng, beta, sigma, gamma):
        # method to update the SEIR states using SEIR probabilities specified

        old_state = self.state

        if self.state == Agent.SUSCEPTIBLE:
            if self.infected_neighbour(lattice):
                if rng.random() < beta:
                    self.state = Agent.EXPOSED

        elif self.state == Agent.EXPOSED:
            if rng.random() < sigma:
                self.state = Agent.INFECTED

        elif self.state == Agent.INFECTED:
            if rng.random() < gamma:
                self.state = Agent.RECOVERED

        if self.state != old_state:
            lattice[self.xpos, self.ypos] = self.state




    
   






