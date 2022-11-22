### This file is used to perform multiple evolution runs at once ###
import timeit
import random
from funcs import *
starttimer0 = timeit.default_timer()
random.seed(0) # set random generators for repeatability
np.random.seed(0)

# change probability of mutation
settings0 = [[24,3,6,2,0.8,0.4,2,0,[7000,-2000],2,100], 
             [25,3,6,2,0.6,0.4,2,0,[7000,-2000],2,100],
             [26,3,6,2,0.4,0.4,2,0,[7000,-2000],2,100],
             [27,3,6,2,0.2,0.4,2,0,[7000,-2000],2,100]]

# change best individuals
settings1 = [[28,10,100,50,0.4,0.4,2,0,[7000,-2000],2,100], 
             [29,10,100,25,0.4,0.4,2,0,[7000,-2000],2,100],
             [30,10,100,10,0.4,0.4,2,0,[7000,-2000],2,100],
             [31,10,100,5,0.4,0.4,2,0,[7000,-2000],2,100],
             [32,10,100,2,0.4,0.4,2,0,[7000,-2000],2,100]]

# change binary/ternary input
settings2 = [[33,20,10,4,0.4,0.4,1,0,[7000,-2000],2,100], # binary
             [34,20,10,4,0.4,0.4,1,0,[7000,-2000],2,100], # binary
             [35,20,10,4,0.4,0.4,2,0,[7000,-2000],2,100], # ternary
             [36,20,10,4,0.4,0.4,2,0,[7000,-2000],2,100]] # ternary

# change probability of mutation
settings3 = [[39,10,25,5,0.8,0.4,2,0,[7000,-2000],2,100], 
             [40,10,25,5,0.6,0.4,2,0,[7000,-2000],2,100],
             [41,10,25,5,0.4,0.4,2,0,[7000,-2000],2,100],
             [42,10,25,5,0.2,0.4,2,0,[7000,-2000],2,100]]

for i in settings3:
    starttimer1 = timeit.default_timer()
    print('EVOLUTION: ', i[0])
    run_evolution(i)
    stoptimer1 = timeit.default_timer()
    print('EVOLUTION DONE IN: ', round(stoptimer1-starttimer1,2))

stoptimer0 = timeit.default_timer()
print('Time: ', round(stoptimer0-starttimer0,2))
