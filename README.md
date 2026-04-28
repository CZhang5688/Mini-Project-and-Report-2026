# Mini Project README
## What is contained in this Git Bundle
This git bundle comprises of two python files. 
1) uu24101_part_1.py - This python script contains code that when run, produces an SEIR model which is modelled as an Initial Value problem.
   This is written using imperative programming. ( This will be referred to as the 'part 1 file')
2) uu24101_part_2.py - This python script contains code that when run, is able to produce Monte Carlo SEIR models. (This will be
   referred to as the 'part 2 file')


## Dependencies 
These files depend on the following installation requirements:
- Python
- numpy
- matplotlib
- scipy
  
## How to run these files
1) To run simulations in the part 1 file, simply call the 'RUN_SEIR_Simulation' function and pass in your desired conditions
and initials for your model into its parameters.


3) To run the simulations for the part 2 file, create an MonteCarlo_SEIRSimulation object and pass in your desired conditions
into its attributes, and then use the 'run' method to run the simulation, and then use the 'plot_results' method on the object to produce a
2D lattice plot and a population plot against the monte-carlo steps of your simulation.

Since these are both python scripts, no compilation or executables are required to run these files at all. These files can simply each be run
as standalone files. 


## Testing
This code was testing by comparing default models to example models provided by an assesssment brief which these scripts intended to solve.



