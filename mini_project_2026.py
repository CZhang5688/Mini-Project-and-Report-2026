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
    
    time_points = np.linspace(start_time, end_time, 1000)

    results = solve_ivp(SEIR_equations, [start_time, end_time], initial_conditions,
                        args=(infection_rate, incubation_rate, recovery_rate),
                        t_eval=time_points)

    t = results.t
    s = results.y[0]
    e = results.y[1]
    i = results.y[2]
    r = results.y[3]

    plt.figure()
    plt.plot(t, s, label="Susceptible")
    plt.plot(t, e, label="Exposed")
    plt.plot(t, i, label="Infected")
    plt.plot(t, r, label="Recovered")
    plt.xlabel("Time")
    plt.ylabel("Fraction of population")
    plt.title(title)
    plt.legend()
    plt.show()

    return results

run_SEIR_Simulation()
