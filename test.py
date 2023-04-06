import gym
import numpy as np
import pickle
import math








# print number of successes for each agent
num_agent=11
for  i in range(num_agent):
    with open("results/rrt option/agent"+str(i+1)+"/locations", 'rb') as fp:
                locations=pickle.load(fp)
    success=0
    for k in range(len(locations)):
        if k>8e6:
              continue
        location=locations[k]
        if math.sqrt(math.pow(location[0]-8.8503,2)+math.pow(location[1]-9.1610,2))<=0.15:
            success+=1
    print("agent"+str(i+1)+":   "+str(success))
    print("frames agent"+str(i+1)+":   "+str(len(locations)))
    print("*"*30)



