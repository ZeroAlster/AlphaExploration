import gym
import numpy as np
import pickle
import math








for i in range(6):
    if i==3:
        continue

    with open("results/rrt option/agent"+str(i+1)+"/locations", 'rb') as fp:
                destinations=pickle.load(fp)

    success=0
    for destination in destinations:
        if math.sqrt(math.pow(destination[0]-8.8503,2)+math.pow(destination[1]-9.1610,2))<0.15:
             success+=1

    print("agent "+ str(i+1)+"  succeeded: "+str(success)) 



