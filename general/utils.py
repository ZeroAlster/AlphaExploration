import gym
import numpy as np
import pickle
import math
# from agent import Actor
# from agent import Critic
import torch
import matplotlib.pyplot as plt
import statistics
from matplotlib.font_manager import FontProperties


# success1=[]
# success2=[]
# success3=[]
# address1="our_method/results/maze/full"
# address2="our_method/results/maze/update/one_step_TD"
# address3="our_method/results/maze/update/avg8_step_TD"

# for i in range(11):
#     with open(address1+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success1.append(pickle.load(fp))
# for i in range(9):
#     with open(address2+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success2.append(pickle.load(fp))
# for i in range(6):
#     with open(address3+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success3.append(pickle.load(fp))

# plt.figure()
# ax = plt.subplot(111)
# all_success=[]
# labels=["our method","one-step TD","MMDDPG(8-avg)"]
# colors=["blue","darkorange","green"]
# all_success.append(success1)
# all_success.append(success2)
# all_success.append(success3)


# for k in range(3):
#     number=len(success2[0])
#     std=np.zeros((1,number))
#     mean=np.zeros((1,number))
#     horizon=np.zeros((1,number))

#     for i in range(number):
#         values=[]
#         agents=6
#         if k==0:
#             agents=11
#         elif k==1:
#             agents=9
#         for j in range(agents):
#             values.append(all_success[k][j][i])
#         mean[0][i]=sum(values)/len(values)
#         std[0][i]=statistics.pstdev(values)
#         horizon[0][i]=i
    
#     plt.plot(horizon[0,:],mean[0,:], 'k-',color=colors[k],label=labels[k])

#     # fix the error bar
#     std=std
#     down_bar=np.maximum((mean-std)[0,:],0)
#     up_bar=np.minimum((mean+std)[0,:],1)

#     plt.fill_between(horizon[0,:],down_bar,up_bar,color=colors[k],alpha=0.2)

# fontP = FontProperties()
# fontP.set_size('x-small')

# plt.title("success rate")
# plt.xlabel("checkpoints")
# ax.legend(loc="upper left",prop=fontP)
# plt.savefig("general/final_figures/update.png")




# plot the success rates for the paper
######################################
# success1=[]
# success2=[]
# success3=[]
# success4=[]
# success5=[]
# address1="DDPG/results/results_maze"
# address2="DDPG_HER/results/results_maze"
# address3="DDPG_ICM/results/results_maze"
# address4="our_method/results/results_maze/full_results"
# address5="SAC/results/results_maze"

# for i in range(10):
#     with open(address1+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success1.append(pickle.load(fp))
# for i in range(5):
#     with open(address2+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success2.append(pickle.load(fp))
# for i in range(5):
#     with open(address3+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success3.append(pickle.load(fp))
# for i in range(11):
#     with open(address4+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success4.append(pickle.load(fp))
# for i in range(10):
#     with open(address5+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success5.append(pickle.load(fp))

# plt.figure()
# ax = plt.subplot(111)
# all_success=[]
# labels=["DDPG","DDPG + HER","DDPG + intrinsic motivation","our method","SAC"]
# colors=["blue","darkorange","green","purple","aqua"]
# all_success.append(success1)
# all_success.append(success2)
# all_success.append(success3)
# all_success.append(success4)
# all_success.append(success5)


# for k in range(5):
#     number=len(success1[0])
#     std=np.zeros((1,number))
#     mean=np.zeros((1,number))
#     horizon=np.zeros((1,number))

#     for i in range(number):
#         values=[]
#         agents=5
#         if k==0 or k==4:
#             agents=10
#         elif k==3:
#             agents=11
#         for j in range(agents):
#             values.append(all_success[k][j][i])
#         mean[0][i]=sum(values)/len(values)
#         std[0][i]=statistics.pstdev(values)
#         horizon[0][i]=i
    
#     plt.plot(horizon[0,:],mean[0,:], 'k-',color=colors[k],label=labels[k])

#     # fix the error bar
#     std=std
#     down_bar=np.maximum((mean-std)[0,:],0)
#     up_bar=np.minimum((mean+std)[0,:],1)

#     plt.fill_between(horizon[0,:],down_bar,up_bar,color=colors[k],alpha=0.2)

# fontP = FontProperties()
# fontP.set_size('x-small')

# plt.title("success rate")
# plt.xlabel("checkpoints")
# ax.legend(loc="upper left",prop=fontP)
# plt.savefig("general/final_figures/success_rates.png")
#####################################


# plot environment coverage for the paper
# #####################################
# explorations1=[]
# explorations2=[]
# explorations3=[]
# explorations4=[]
# address1="DDPG_ICM/results/results_maze"
# address2="DDPG_temporal/results/results_maze/results"
# address3="DDPG/results/results_maze"
# address4="our_method/results/results_maze/full_results"

# for i in range(5):
#     with open(address1+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
#             explorations1.append(pickle.load(fp))
# for i in range(5):
#     with open(address2+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
#             explorations2.append(pickle.load(fp))
# for i in range(10):
#     with open(address3+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
#             explorations3.append(pickle.load(fp))
# for i in range(11):
#     with open(address4+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
#             explorations4.append(pickle.load(fp))

# plt.figure()
# ax = plt.subplot(111)
# explorations=[]
# labels=["DDPG + intrinsic motivation","DDPG + \u03B5z-Greedy ","DDPG","our method"]
# colors=["blue","darkorange","green","purple"]
# explorations.append(explorations1)
# explorations.append(explorations2)
# explorations.append(explorations3)
# explorations.append(explorations4)

# for k in range(4):
#     number=100
#     std=np.zeros((1,number))
#     mean=np.zeros((1,number))
#     horizon=np.zeros((1,number))

#     for i in range(number):
#         values=[]
#         agent=5
#         if k==2:
#             agents=10
#         elif k==3:
#              agents=11 

#         for j in range(5):
#             values.append(explorations[k][j][i])
#         mean[0][i]=sum(values)/len(values)
#         std[0][i]=statistics.pstdev(values)
#         horizon[0][i]=i
    
#     plt.plot(horizon[0,:],mean[0,:], 'k-',color=colors[k],label=labels[k])

#     # fix the error bar
#     down_bar=np.maximum((mean-std)[0,:],0)
#     up_bar=np.minimum((mean+std)[0,:],1)

#     plt.fill_between(horizon[0,:],down_bar,up_bar,color=colors[k],alpha=0.2)

# plt.title("environment coverage")
# plt.xlabel("checkpoints")
# ax.legend(loc="upper left")
# plt.savefig("general/final_figures/env_coverage.png")
# ######################################





# print number of successes for each agent
######################################
num_agent=10
for  i in range(num_agent):
    with open("our_method/results/maze/buffer/Two_FIFO/agent"+str(i+1)+"/locations", 'rb') as fp:
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


