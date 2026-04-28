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


class MonteCarlo_SEIRSimulation:
    # class to run the Monte Carlo SEIR simulation
    # default values matched to the example provided in assessment brief 
    # for testing 
    def __init__(
        self,
        numlattice=100,
        num_agents=250,
        p_susceptible=0.95,
        beta=1.0,
        sigma=0.1,
        gamma=0.005,
        nsteps=2000,
        seed=1234,
    ):
        self.numlattice = numlattice
        self.num_agents = num_agents
        self.p_susceptible = p_susceptible
        self.p_exposed = 1.0 - p_susceptible

        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.nsteps = nsteps

        self.rng = np.random.default_rng(seed)

        self.lattice = np.zeros((numlattice, numlattice), dtype=int)
        self.agents = []

        self.susceptible_count = np.zeros(nsteps, dtype=int)
        self.exposed_count = np.zeros(nsteps, dtype=int)
        self.infected_count = np.zeros(nsteps, dtype=int)
        self.recovered_count = np.zeros(nsteps, dtype=int)

        self.initialise_agents()

    def initialise_agents(self):
    # method to place the agents randomly on the 2D lattice initially. 
        for i in range(self.num_agents):
            placed = False

            while not placed:
                x = self.rng.integers(0, self.numlattice)
                y = self.rng.integers(0, self.numlattice)

                if self.lattice[x, y] == Agent.EMPTY:

                    state = self.rng.choice(
                        [Agent.SUSCEPTIBLE, Agent.EXPOSED],
                        p=[self.p_susceptible, self.p_exposed]
                    )

                    agent = Agent((x, y), state)

                    self.agents.append(agent)
                    self.lattice[x, y] = state

                    placed = True

        self.count_compartments(0)

    def count_compartments(self, step):
    # method to count how many agents are in each SEIR compartment
        states = [agent.state for agent in self.agents]

    # counting how many agents are in each different state
        self.susceptible_count[step] = states.count(Agent.SUSCEPTIBLE)
        self.exposed_count[step] = states.count(Agent.EXPOSED)
        self.infected_count[step] = states.count(Agent.INFECTED)
        self.recovered_count[step] = states.count(Agent.RECOVERED)

    def step(self, step_number):
    # method to perform one monte carlo step
        for agent in self.agents:
            agent.move(self.lattice, self.rng)

            agent.update_state(
                self.lattice,
                self.rng,
                self.beta,
                self.sigma,
                self.gamma,
            )

        # counting the number of agents in each state at each step 
        self.count_compartments(step_number)


    def plot_lattice(self):
        # method to plot the resulting lattice configuration from the simulation 

        # assigning the colours for each state 
        colours = {
            Agent.SUSCEPTIBLE: "blue",
            Agent.EXPOSED: "orange",
            Agent.INFECTED: "red",
            Agent.RECOVERED: "green"
        }

        # labeling each state for plotting 
        labels = {
            Agent.SUSCEPTIBLE: "Susceptible",
            Agent.EXPOSED: "Exposed",
            Agent.INFECTED: "Infected",
            Agent.RECOVERED: "Recovered"
        }

        # creating a new plot window 
        plt.figure()

        for state in [Agent.SUSCEPTIBLE, Agent.EXPOSED, Agent.INFECTED, Agent.RECOVERED]:
            xs = [agent.xpos for agent in self.agents if agent.state == state]
            ys = [agent.ypos for agent in self.agents if agent.state == state]

            plt.scatter(xs, ys, s=20, c=colours[state], label=labels[state])


        # plotting results 
        plt.xlim(0, self.numlattice)
        plt.ylim(0, self.numlattice)
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Monte Carlo SEIR simulation")
        plt.legend()
        plt.show()

    def plot_population(self):
        # method to plot the population against the Monte Carlo

        steps = np.arange(self.nsteps)

        # plotting the population against the Monte Carlo Steps 
        plt.figure(figsize=(9, 6))
        plt.plot(steps, self.susceptible_count, label="Susceptible")
        plt.plot(steps, self.exposed_count, label="Exposed")
        plt.plot(steps, self.infected_count, label="Infected")
        plt.plot(steps, self.recovered_count, label="Recovered")
        plt.xlabel("Monte Carlo step")
        plt.ylabel("Population")
        plt.title("Monte Carlo SEIR simulation")
        plt.legend()
        plt.show()

    def plot_results(self):
    # simple function to actually plot both the lattice and population plot.  
        self.plot_lattice()
        self.plot_population()

    def run(self):
    # method to run a Monte Carlo SEIR simulation 

        for step in range(1, self.nsteps):
            self.step(step)

            if step % 100 == 0:
                print(
                    f"Step {step}: "
                    f"S={self.susceptible_count[step]}, "
                    f"E={self.exposed_count[step]}, "
                    f"I={self.infected_count[step]}, "
                    f"R={self.recovered_count[step]}"
                )

    
if __name__ == "__main__":

    # initialising and creating the simulation, with default constraints 
    simulation = MonteCarlo_SEIRSimulation(
        numlattice=100,
        num_agents=250,
        p_susceptible=0.95,
        beta=1.0,
        sigma=0.1,
        gamma=0.005,
        nsteps=2000,
        seed=1234,
    )

    # running the simulation 
    simulation.run()
    # plotting the results of the simulation 
    simulation.plot_results()





    


    
   






