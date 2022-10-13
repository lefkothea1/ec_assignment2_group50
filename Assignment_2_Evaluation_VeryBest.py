# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller

import numpy as np
import time
import concurrent.futures
import pandas as pd

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


### testing:

def simulation(env,x): # returns the individuell fitness level for one solution
    f,p,e,t = env.play(pcont=x)
    return f,p,e

def simulation_finalBest(env,x): # returns the individuell fitness level for one solution
    f,p,e,t = env.play(pcont=np.array(x))
    return f,p,e # as a tuple to be iterable


def finalBest(hof, iterations = 5):

    enemies = [1,2,3,4,5,6,7,8]
    player = []
    enemy = []

    for en in enemies:
        #Update the enemy
        env.update_parameter('enemies',[en])

        avg_p = []
        avg_e = []

        for iter in range(iterations):
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                _,p,e = simulation_finalBest(env, hof)

            avg_p.append(p)
            avg_e.append(e)
        
        player.append(np.mean(avg_p))
        enemy.append(np.mean(avg_e))

    
    df = pd.DataFrame(list(zip(player, enemy)), columns = ["player", "enemy"])
    df.to_csv("evaluation_very_best.csv")

    return list(zip(player, enemy))


## Main:

# read in very best solution

# MO-CMA ES, group{7,8}, first run

# read in file
# empty list to read list from a file
hof = []

with open("experiment_advanced_group2_hof.txt", 'r') as fp:
    for ind in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = ind[1:-2]

        # add current item to the list
        hof.append(list(eval(x)))

hof = hof[0]

results = finalBest(hof)


## Bring it in the format to hand it in

hof = np.array(hof)
np.savetxt("Assignment_2_Very_Best_Solution", hof)
