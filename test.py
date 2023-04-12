import gym
import numpy as np
import pickle
import math
from agent import Actor
from agent import Critic
import torch





# print number of successes for each agent
######################################
num_agent=15
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
######################################


# load models of the agent to check values of successfull trajectories
######################################
# actor=Actor(4,128,2,0.95)
# critic=Critic(6,128,1)
# actor.load_state_dict(torch.load('results/rrt option/agent1/actor.pth'))
# critic.load_state_dict(torch.load('results/rrt option/agent1/critic.pth'))
# with open("results/rrt option/agent6/successful_trajectories", 'rb') as fp:
#                 samples=pickle.load(fp)

# print("number of samples: "+str(len(samples)))
# for sample in samples:
#     state=torch.FloatTensor([sample[0]])
#     action=torch.FloatTensor([sample[1]])
#     val = critic.forward(state, action)
#     print("*"*30)
#     print("state: "+str(state))
#     print("action: "+str(action))
#     print("reward: "+str(sample[2]))
#     print("value: "+str(val))
#     input("Press Enter to continue...")



# check amount of exploration for the agent
######################################
# print(math.pow(0.99999975,int(7e6)))
######################################






