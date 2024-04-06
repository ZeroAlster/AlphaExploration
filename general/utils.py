import gym
import numpy as np
import pickle
import math
# from agent import Actor
# from agent import Critic
import torch
import matplotlib.pyplot as plt
import statistics
from scipy.interpolate import make_interp_spline
from matplotlib.font_manager import FontProperties
import matplotlib.colors as cor
import pylab


#full colors and labels
# colors=["darkorange","blue","red","green","purple","saddlebrown","aqua"]
# labels=["DDPG","ETGL-DDPG (perfect model)","ETGL-DDPG (replay buffer)","DDPG + HER","DDPG + \u03B5z-Greedy","DDPG + intrinsic motivation","SAC"]



# the seperate legend if it is needed
# colors=["purple","blue","darkorange","green","red"]
# fig = pylab.figure()
# figlegend = pylab.figure(figsize=(13,0.5))
# ax = fig.add_subplot(111)
# lines = ax.plot(range(10), colors[0], range(10), colors[1], range(10), colors[2], range(10), colors[3], range(10), colors[4])
# figlegend.legend(lines, ("DDPG","DDPG + \u03B5t-Greedy (perfect model)","DDPG + \u03B5t-Greedy (replay buffer)","DDPG + GDRB","DDPG + longest ns-tep return"),loc='center',ncol=5)
# fig.show()
# figlegend.show()
# figlegend.savefig('legend.png',bbox_inches='tight')




# success1=[]
# success2=[]
# address1="our_method/results/push/update/avg8-step-TD"
# address2="our_method/results/push/full (perfect model)"


# for i in range(5):
#     with open(address1+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success1.append(pickle.load(fp))
# for i in range(17):
#     if i==15 or i==16:
#          continue
#     with open(address2+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#             success2.append(pickle.load(fp))


# plt.figure()
# ax = plt.subplot(111)
# all_success=[]
# colors=["blue","red"]
# all_success.append(success1)
# all_success.append(success2)



# for k in range(2):

#     number=len(all_success[0][0])

#     std=[]
#     mean=[]
#     horizon=[]

#     i=0
#     beta=10
#     while i < number:
#         values=[]
#         agents=len(all_success[k])
        
#         for j in range(agents):
#             values.append(all_success[k][j][i])
#         mean.append(sum(values)/len(values))
#         std.append(statistics.pstdev(values))
#         horizon.append(i/beta)

#         i+=beta
    

#     mean=np.array(mean)
#     std=np.array(std)
#     horizon=np.array(horizon)
    
#     #smoothing the plots
#     X_Y_Spline = make_interp_spline(horizon, mean)
#     X_ = np.linspace(horizon.min(), horizon.max(),50)
#     Y_ = X_Y_Spline(X_)
#     Y_=np.minimum(Y_,1)
#     Y_=np.maximum(Y_,0)
#     plt.plot(X_, Y_,color=colors[k])
    
#     # ax.plot(horizon,mean,color=colors[k])

#     # fix the error bar
#     std=std*0.6
#     down_bar=np.maximum((mean-std),0)
#     up_bar=np.minimum((mean+std),1)

#     ax.fill_between(horizon,down_bar,up_bar,color=colors[k],alpha=0.1)

# fontP = FontProperties()
# fontP.set_size('x-small')

# plt.title("success rate")
# plt.xlabel("checkpoints")
# plt.savefig("test")
#####################################




# plot the success rates for the paper
######################################
# success1=[]
# success2=[]
# success3=[]
# success4=[]
# success5=[]
# success6=[]
# success7=[]
# address1="DDPG/results/simple_maze2"
# address2="our_method/results/simple_maze2/perfect model"
# address3="our_method/results/simple_maze2/replay buffer"
# address4="DDPG_HER/results/simple_maze2"
# address5="DDPG_temporal/results/simple_maze2"
# address6="DDPG_ICM/results/simple_maze2"
# address7="SAC/results/simple_maze2"


# for i in range(5):
#     with open(address1+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success1.append(pickle.load(fp))
# for i in range(7):
#     with open(address2+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#             success2.append(pickle.load(fp))
# for i in range(10):
#     bads=[2,3,4,5]
#     if i in bads:
#          continue
#     with open(address3+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success3.append(pickle.load(fp))
# for i in range(6):
#     bads=[0,2,3,4]
#     if i in bads:
#          continue
#     with open(address4+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#             success4.append(pickle.load(fp))
# for i in range(6):
#     with open(address5+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success5.append(pickle.load(fp))
# for i in range(5):
#     with open(address6+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#             success6.append(pickle.load(fp))
# for i in range(7):
#     bads=[4,5]
#     if i in bads:
#          continue
#     with open(address7+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
#         success7.append(pickle.load(fp))

# plt.figure()
# ax = plt.subplot(111)
# all_success=[]
# # labels=["DDPG","ETGL-DDPG ()","ETGL-DDPG ()","DDPG + HER","DDPG + "]
# colors=["darkorange","blue","red","green","purple","saddlebrown","aqua"]
# all_success.append(success1)
# all_success.append(success2)
# all_success.append(success3)
# all_success.append(success4)
# all_success.append(success5)
# all_success.append(success6)
# all_success.append(success7)


# for k in range(7):

#     noisy=[1,2,4]
#     number=len(all_success[0][0])

#     std=[]
#     mean=[]
#     horizon=[]

#     i=0
#     beta=10
#     while i < number:
#         values=[]
#         agents=len(all_success[k])
        
#         for j in range(agents):
#             values.append(all_success[k][j][i])
#         mean.append(sum(values)/len(values))
#         std.append(statistics.pstdev(values))
#         horizon.append(i/beta)

#         i+=beta
    

#     mean=np.array(mean)
#     std=np.array(std)
#     horizon=np.array(horizon)
    
#     #smoothing the plots
#     X_Y_Spline = make_interp_spline(horizon, mean)
#     if k in noisy:
#         X_ = np.linspace(horizon.min(), horizon.max(),20)
#     else:     
#         X_ = np.linspace(horizon.min(), horizon.max(),80)
#     Y_ = X_Y_Spline(X_)
#     Y_=np.minimum(Y_,1)
#     Y_=np.maximum(Y_,0)
#     plt.plot(X_, Y_,color=colors[k])
    
#     # ax.plot(horizon,mean,color=colors[k])

#     # fix the error bar
#     std=std*0.6
#     down_bar=np.maximum((mean-std),0)
#     up_bar=np.minimum((mean+std),1)

#     ax.fill_between(horizon,down_bar,up_bar,color=colors[k],alpha=0.1)

# fontP = FontProperties()
# fontP.set_size('x-small')

# plt.title("success rate")
# plt.xlabel("checkpoints")
# plt.savefig("test")
###################################


# plot environment coverage for the paper
#####################################
# explorations1=[]
# explorations2=[]
# explorations3=[]
# explorations4=[]
# explorations5=[]
# address1="DDPG_ICM/results/push"
# address2="DDPG_temporal/results/push/full"
# address3="DDPG/results/push"
# address4="our_method/results/push/full (perfect model)"
# address5="our_method/results/push/full (replay buffer)"

# for i in range(10):
#     with open(address1+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
#             explorations1.append(pickle.load(fp))
# for i in range(10):
#     with open(address2+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
#             explorations2.append(pickle.load(fp))
# for i in range(10):
#     with open(address3+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
#             explorations3.append(pickle.load(fp))
# for i in range(10):
#     with open(address4+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
#             explorations4.append(pickle.load(fp))
# for i in range(10):
#     with open(address5+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
#             explorations5.append(pickle.load(fp))

# plt.figure()
# ax = plt.subplot(111)
# explorations=[]
# labels=["DDPG + intrinsic motivation","DDPG + \u03B5z-Greedy ","DDPG","DDPG + \u03B5t-greedy (perfect model)", "DDPG + \u03B5t-greedy (replay buffer)"]
# colors=["darkorange","blue","red","green","purple","gold","aqua"]
# explorations.append(explorations1)
# explorations.append(explorations2)
# explorations.append(explorations3)
# explorations.append(explorations4)
# explorations.append(explorations5)

# for k in range(5):
#     number=600
#     std=np.zeros((1,number))
#     mean=np.zeros((1,number))
#     horizon=np.zeros((1,number))
#     for i in range(number):
#         values=[]
#         agents=10
#         # if k==0:
#         #     agents=5
#         # elif k==1:
#         #      agents=5
#         # elif k==4:
#         #      agents=12 

#         for j in range(agents):
#             # for mujoco
#             # values.append(explorations[k][j][i]*(25/17))
#             # for maze
#             # values.append(explorations[k][j][i])
#             # for push
#             values.append(explorations[k][j][i]*(1.09))
#         mean[0][i]=sum(values)/len(values)
#         std[0][i]=statistics.pstdev(values)
#         horizon[0][i]=i
    
#     plt.plot(horizon[0,:],mean[0,:],color=colors[k],label=labels[k])

#     # fix the error bar
#     down_bar=np.maximum((mean-std)[0,:],0)
#     up_bar=np.minimum((mean+std)[0,:],1)

#     plt.fill_between(horizon[0,:],down_bar,up_bar,color=colors[k],alpha=0.2)

# plt.title("environment coverage")
# plt.xlabel("checkpoints")
# plt.yticks([0,0.2,0.4,0.6,0.8,1])
# plt.gca().set_ylim(top=1.05)
# # ax.legend(loc="lower right")
# plt.savefig("test")
# ######################################




# print number of successes for each agent
######################################
num_agent=10
for  i in range(num_agent):
    with open("PPO/results/mujoco/agent"+str(i+1)+"/locations", 'rb') as fp:
                locations=pickle.load(fp)
    success=0
    for k in range(len(locations)):
        location=locations[k]
        if math.sqrt(math.pow(location[0]-0,2)+math.pow(location[1]-16,2))<=0.6:
        # if math.sqrt(math.pow(location[0]-8.8503,2)+math.pow(location[1]-9.1610,2))<=0.15:
        # if math.sqrt(math.pow(location[0]-4,2)+math.pow(location[1]-24.8,2))<=0.6:
            success+=1
    print("agent"+str(i+1)+":   "+str(success))
    print("frames agent"+str(i+1)+":   "+str(len(locations)))
    print("*"*30)
######################################

