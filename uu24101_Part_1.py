# importing the required libraries for this project
import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# creating the monte carlo simulation function 
def monte(myfunc, xmin, xmax, N):
    samples = np.random.uniform(low=xmin, high=xmax, size=N)
    values = myfunc(samples)
    mean = values.sum()/N
    meansq = (values*values).sum()/N
    integral = (xmax - xmin) * mean
    error = (xmax - xmin) * np.sqrt((meansq - mean * mean) / N)
    return (integral, error)

# PART 1
# creating the function to solve the coupled set of equations (7) - (10)
# (infection_rate = beta, incubation_rate = sigma, recovery_rate = gamma) 

def SEIR_equations(t, y, infection_rate, incubation_rate, recovery_rate):
    s = y[0]
    e = y[1]
    i = y[2]
    r = y[3]
    dsdt = -infection_rate * i * s
    dedt = (infection_rate * i * s) - (incubation_rate * e)
    didt = (incubation_rate * e) - (recovery_rate * i)
    drdt = recovery_rate * i 

    return [dsdt, dedt, didt, drdt]


# verifying its working by plotting results and comparing the results with the figure provided 
# definining initial conditions 

def run_SEIR_Simulation(title = "default",
                        initial_conditions = [0.99, 0.01, 0.0, 0.0], 
                        start_time = 0,
                        end_time = 100,
                        infection_rate = 1.0,
                        incubation_rate = 1.0,
                        recovery_rate = 0.1):
    
    # creating the time points to plot the graph for each day
    time_points = np.linspace(start_time, end_time, 1000)

    # solving the SEIR equations at each time step and results of the SEIR model for every day
    results = solve_ivp(SEIR_equations, [start_time, end_time], initial_conditions,
                        args=(infection_rate, incubation_rate, recovery_rate),
                        t_eval=time_points)

    # storing the results in separate named variables for easier debugging and 
    # tracing of each variable
    t = results.t
    s = results.y[0]
    e = results.y[1]
    i = results.y[2]
    r = results.y[3]

    # plotting the results 
    plt.figure()
    plt.plot(t, s, label="Susceptible")
    plt.plot(t, e, label="Exposed")
    plt.plot(t, i, label="Infected")
    plt.plot(t, r, label="Recovered")
    plt.xlabel("Time (days)")
    plt.ylabel("Fraction of population")
    plt.title(title)
    plt.legend()
    plt.show()

    return results


# running the default simulation to test it and compare it against the example provided
run_SEIR_Simulation()

# running simulation with a doubled infection rate
run_SEIR_Simulation(title = "Doubled_infection rate", infection_rate = 2.0)

# running simulation with a higher initial exposed fraction of population
run_SEIR_Simulation(title = "higher_initial_exposed", initial_conditions = [0.50, 0.50, 0.0, 0.0])

# running simulation with a 1/10 lower recovery rate
run_SEIR_Simulation(title = "low_initial_recovery_rate", recovery_rate = 0.01)

# running simulation with a 1/10 lower recovery rate and longer end time. 
run_SEIR_Simulation(title = "low_initial_recovery_rate", recovery_rate = 0.01, end_time = 500)

# running a simulation with both different initial values and different transition rates
run_SEIR_Simulation(title= "multiple different parameters",initial_conditions=[0.8,0.2,0.0,0.0], 
                    recovery_rate = 0.05, incubation_rate = 1.5, infection_rate = 2.0)


