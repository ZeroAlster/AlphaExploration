import gym
import numpy as np
import pickle
import math
# from agent import Actor
# from agent import Critic
import torch
import matplotlib.pyplot as plt




# print number of successes for each agent
######################################
num_agent=10
for  i in range(num_agent):
    with open("DDPG_ICM/results/agent"+str(i+1)+"/locations", 'rb') as fp:
                locations=pickle.load(fp)
    success=0
    for k in range(len(locations)):
        location=locations[k][0]
        if math.sqrt(math.pow(location[0]-8.8503,2)+math.pow(location[1]-9.1610,2))<=0.15:
            success+=1
    print("agent"+str(i+1)+":   "+str(success))
    print("frames agent"+str(i+1)+":   "+str(len(locations)))
    print("*"*30)
######################################


# load models of the agent to check values of successfull trajectories
######################################
# actor=Actor(4,256,2,0.95)
# critic=Critic(6,256,1)
# actor.load_state_dict(torch.load('results/rrt option/agent6/actor.pth'))
# critic.load_state_dict(torch.load('results/rrt option/agent6/critic.pth'))
# with open("results/rrt option/agent6/successful_trajectories", 'rb') as fp:
#                 samples=pickle.load(fp)


# # # some random samples
# state=torch.FloatTensor([[8.2811, 9.0004, 8.8503, 9.1610]])
# action=torch.FloatTensor([[0.0,0.0]])
# val = critic.forward(state, action)
# print("*"*30)
# print("state: "+str(state))
# print("action: "+str(action))
# print("value: "+str(val))

# state=torch.FloatTensor([[7.23,1.45,8.8503, 9.1610]])
# action=torch.FloatTensor([[0.21,-0.98]])
# val = critic.forward(state, action)
# print("*"*30)
# print("state: "+str(state))
# print("action: "+str(action))
# print("value: "+str(val))



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
######################################


# check amount of exploration for the agent
######################################
# print(math.pow(0.9999988,int(10e6)))
######################################


# see when we get to the goal during training
######################################
# with open("results/rrt option/agent1/locations", 'rb') as fp:
#     locations=pickle.load(fp)

# success=0
# successes=[]
# for k in range(len(locations)):
#     location=locations[k]
#     if math.sqrt(math.pow(location[0]-8.8503,2)+math.pow(location[1]-9.1610,2))<=0.15:
#         success+=1
#     successes.append(success)

# plt.plot(successes)
# plt.savefig("test")
######################################


