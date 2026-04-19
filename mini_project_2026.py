# importing the required libraries for this project
import numpy as np 

# creating the monte carlo simulation function 
def monte(myfunc, xmin, xmax, N):
    samples = np.random.uniform(low=xmin, high=xmax, size=N)
    values = myfunc(samples)
    mean = values.sum()/N
    meansq = (values*values).sum()/N
    integral = (xmax - xmin) * mean
    error = (xmax - xmin) * np.sqrt((meansq - mean * mean) / N)
    return (integral, error)


