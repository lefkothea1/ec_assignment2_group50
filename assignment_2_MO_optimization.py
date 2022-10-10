################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

## Get used to the framework, first steps


# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller

from deap import base, creator, tools, algorithms, cma

import time
import numpy as np
import random

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# hidden neurons in the network we want to evolve!
n_hidden_neurons = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name, 
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

                  
# default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

ini = time.time()  # sets time marker

# number of weights for multilayer with 10 hidden neurons - this is the "genotype representation" of each individuel
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


# parameters for a genetic algorithm
upper = 1
lower = -1
lambda_ = 20
mu = int(lambda_ / 2)


# initialize group of enemies
group = [7,8]
#group = [1,5,6]

# Fitness functions

def simulation(env,x): # returns the individuell fitness level for one solution
    f,p,e,t = env.play(pcont=x)
    return f,p,e

# but in DEAP its needed for single individuals

# advanced, second objective on beating enemies
# every win +1 
def fitnessFunction_advanced(ind):
    fitness = []
    wins = 0
    # group is needed to be set beforehand:
    for en in group:
        env.update_parameter('enemies',[en])
        f, p, e = simulation(env, np.array(ind))
        if p > e:
            wins += 1 # add extra value to really beat enemies
        fitness.append(f)

    fitness = np.mean(fitness)
    print("Combined fitness:", fitness, "; Number of wins:", wins)
    return (fitness, wins) # return as tuple for DEAP


# DEAP setup

# maximation, one objective
creator.create("FitnessMax", base.Fitness, weights=(0.5,0.5))
creator.create("Individual", list, fitness=creator.FitnessMax)

# set up individuals, random values between -1 and 1
toolbox = base.Toolbox()
toolbox.register("attribute", np.random.uniform, lower, upper)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=n_vars) # n_vars = individuals size
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitnessFunction_advanced)

# Decoration: keeps individuals within the limits
# in the crossover and mutation process
def limit(lower, upper):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > upper:
                        child[i] = upper
                    elif child[i] < lower:
                        child[i] = lower
            return offspring
        return wrapper
    return decorator



stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max, axis=0)
stats.register("avg", np.mean, axis=0)
stats.register("min", np.min, axis=0)
stats.register("std", np.std, axis=0)


## for multi-objective it needs a starting population
# of size mu, with fitness values
population = toolbox.population(n=mu)

for ind in population:
    ind.fitness.values = toolbox.evaluate(ind)

## CMA - ES

def main():


    strategy = cma.StrategyMultiObjective(population, sigma=1.0, mu=mu, lambda_=lambda_)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # set decoration
    toolbox.decorate("generate", limit(lower, upper))

    # log and hof
    log = tools.Logbook()
    log.header = "gen", "new", "max", "mean", "min", "std"
    hof = tools.HallOfFame(1)

    # algo
    pop, log = algorithms.eaGenerateUpdate(toolbox, ngen=3, stats=stats, halloffame=hof)

    return pop, log, hof

# just testing whether everything works
# pop, log, hof = main()

def optimize():
    
    sigma = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    results = []

    for s in sigma: 
        strategy = cma.StrategyMultiObjective(population, sigma=s, mu=mu, lambda_=lambda_)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        # set decoration
        toolbox.decorate("generate", limit(lower, upper))

        # log
        log = tools.Logbook()
        log.header = "gen", "new", "max", "mean", "min", "std"

        # algo
        _, log = algorithms.eaGenerateUpdate(toolbox, ngen=10, stats=stats)

        # mean of the last generation as performance measure
        results.append([s,log.select("avg")[-1][0], log.select("avg")[-1][1]])

    max_f = 0
    max_w = 0
    best_f = []
    best_w = []
    for setting in results:
        if setting[1] > max_f:
            max_f = setting[1]
            best_f = setting
        if setting[2] > max_w:
            max_w = setting[2]
            best_w = setting

    return best_f, best_w, results

best_f, best_w, results = optimize()

## change direcetory
with open(r'C:\Users\oleh\Documents\Uni\15_Semester\Evolutionary_Computing\Assignments\evoman_framework-master\evoman_framework-master\optimization_2_advanced.txt', 'w') as fp:
    fp.write("%s\n" % best_f)
    fp.write("%s\n" % best_w)
    fp.write("\n")
    for line in results:
        # write each item on a new line
        fp.write("%s\n" % line)
    print('Done')
