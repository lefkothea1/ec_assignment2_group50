# This is the repository of group 50 for Evolutionary Computing 2nd assignment: evolving a generalist agent

0) 50.txt contains the very best solution for the competition. 50.pdf contains the report.

1) files "..._optimization.py" were run first to define the best centroid and sigma values to be used, by structured gridsearch

2) file "assignment2_experiment_classic.py" has the experiment using SO-CMA evolutionary strategy
  file "assignment2_experiment_advanced.py" has the experiment using the MO-CMA evolutionary strategy
  
3) in data folder all generated data can be found
   experimental output consists of 3 files per algorithm:
   
      -"...._results.csv" :  fitness values over the evolutionary process. For the MO algorithm fitness and number of wins is exported per generation.
      
      -"...._hof.csv" : best solution of each of the 10 independent experimental runs
      
      -"...._final.csv" : mean gain and number of wins  of best soltion for each of the 10 independent experimental runs
     
4) Finally plotted results can be found in the 'plot' folder, along with code to replicate them from the data
