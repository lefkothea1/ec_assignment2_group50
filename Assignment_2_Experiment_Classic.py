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

import pandas as pd
import time
import numpy as np
import random
import concurrent.futures


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
def fitnessFunction_classic(ind):
    fitness = []
    # group is needed to be set beforehand:
    for en in group:
        env.update_parameter('enemies',[en])
        f, *_ = simulation(env, np.array(ind))
        fitness.append(f)
    
    fitness = np.mean(fitness)
    print("Combined fitness:", fitness)
    return (fitness,) # return as tuple for DEAP


# DEAP setup

# maximation, one objective
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# set up individuals, random values between -1 and 1
toolbox = base.Toolbox()
toolbox.register("evaluate", fitnessFunction_classic)

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


## MO - CMA - ES

# adapt sigam according to optimization
def CMA_ES(centroid = [0]*n_vars, sigma = 0.1, gen = 30, lambda_ = 100):

    strategy = cma.Strategy(centroid=centroid, sigma=sigma, lambda_ = lambda_)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # set decoration
    toolbox.decorate("generate", limit(lower, upper))

    # log and hof
    log = tools.Logbook()
    log.header = "gen", "new", "max", "mean", "min", "std"
    hof = tools.HallOfFame(1)

    # algo
    with concurrent.futures.ProcessPoolExecutor() as executor:
        pop, log = algorithms.eaGenerateUpdate(toolbox, ngen=gen, stats=stats, halloffame=hof)

    return pop, log, hof



# experiment with 10 times each algorithm
def experiment(function, runs = 10):
    
    bigDF = pd.DataFrame()
    bigHOF = [] # should be in the end of size 10

    for i in range(runs):
        _, log, hof = function()
        df = pd.DataFrame(log)
        bigDF = pd.concat([bigDF, df], ignore_index=True)
        bigHOF.append(hof)
        print("!!!!! NEW ITERATION !!!!!")
        print(i+1)
        print("!!!!! HELLO EVERYBODY I AM VERY HARD TO SPOT !!!!!")
    
    # change name accordingly
    bigDF.to_csv("experiment_classic_group"+str(len(group))+"_results.csv")
    # !
    with open('experiment_classic_group'+str(len(group))+'_hof.txt', 'w') as fp:
        for line in bigHOF:
            # write each item on a new line
            fp.write("%s\n" % line)

    return bigDF, bigHOF


def simulation_finalBest(env,x): # returns the individuell fitness level for one solution
    f,p,e,t = env.play(pcont=np.array(x))
    return f,p,e # as a tuple to be iterable

def finalBest(hof, iterations = 5, runs = 10):
    
    enemies = [1,2,3,4,5,6,7,8]
    gains = []
    wins = []


    for solution in range(runs):

        gain = []
        win = []

        for en in enemies:
        #Update the enemy
            env.update_parameter('enemies',[en])

            avg_g = []
            avg_w = []
            for k in range(iterations):
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    _,p,e = simulation_finalBest(env, hof[solution][0])
                avg_g.append(p-e)
                if p > e:
                    avg_w.append(1)
                else:
                    avg_w.append(0)

            gain.append(np.mean(avg_g))
            win.append(np.mean(avg_w))
        
        gains.append(np.sum(gain))
        wins.append(np.sum(win))

    df = pd.DataFrame(list(zip(gains, wins)),  columns = ["Gain", "Wins"])
    df.to_csv("experiment_classic_group"+str(len(group))+"_final.csv")
        
    return gains, wins


## Main:

# To DO:
# - change directory names in experiment() and final_best() at the end
# - check settings, change group accordingly in the beginning

bigDF, bigHOF = experiment(CMA_ES)

gains, wins = finalBest(bigHOF)



