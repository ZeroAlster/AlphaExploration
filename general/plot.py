import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
import statistics
import random
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import argparse
import sys




def neighbour(state,goal):
        if math.sqrt(math.pow(state[0]-goal[0],2)+math.pow(state[1]-goal[1],2))<=0.6:
            return True
        else:
            return False


def plot(address,success_rates,locations,explorations,num_agents):
    
    # plot success rates for each agent
    for i in range(len(success_rates)):
        plt.figure()
        plt.plot(success_rates[i])
        plt.savefig(address+"/agent"+str(i+1)+"/success_rates")
    

    # # plot success rates all together
    number=len(success_rates[0])
    std=np.zeros((1,number))
    mean=np.zeros((1,number))
    horizon=np.zeros((1,number))
    
    # we plot the results until 6M frames
    for i in range(number):
        values=[]
        for j in range(num_agents):
            values.append(success_rates[j][i])
        mean[0][i]=sum(values)/len(values)
        std[0][i]=statistics.pstdev(values)
        horizon[0][i]=i
    
    plt.figure()
    plt.plot(horizon[0,:],mean[0,:],color="blue")

    # fix the error bar
    std=std
    down_bar=np.maximum((mean-std)[0,:],0)
    up_bar=np.minimum((mean+std)[0,:],1)

    plt.fill_between(horizon[0,:],down_bar,up_bar)
    plt.savefig(address+"/success_rates.png")
    plt.close("all")


    #plot the exploration curves
    # number=len(explorations[7])
    # std=np.zeros((1,number))
    # mean=np.zeros((1,number))
    # horizon=np.zeros((1,number))

    # for i in range(number):
    #     values=[]
    #     for j in range(num_agents):
    #         values.append(explorations[j][i]*(25/17))
    #     mean[0][i]=sum(values)/len(values)
    #     std[0][i]=statistics.pstdev(values)
    #     horizon[0][i]=i
    
    # plt.figure()
    # plt.plot(horizon[0,:],mean[0,:], 'k-',color="blue")
    
    # # fix the error bar
    # down_bar=np.maximum((mean-std)[0,:],0)
    # up_bar=np.minimum((mean+std)[0,:],1)

    # plt.fill_between(horizon[0,:],down_bar,up_bar)
    # plt.savefig(address+"/env_coverage.png")



    
    # define structure of the maze
#     walls=[[(-6,-6),(4,28)],[(-6,-6),(28,4)],[(-6,18),(28,4)],[(18,-6),(4,28)],[(-6,2),(16,8)]]
#     point=[(0,0),1]
#     goal=[(0,16),0.6]

#     # extra blocks
#     walls.append([(2,14),(4,4)])
#     walls.append([(14,10),(4,4)])    
    
#     for i in range(len(locations)):

#         # plot the maze and goal first
#         _, ax = plt.subplots()
#         for wall in walls:
#             ax.add_patch(Rectangle((wall[0][0], wall[0][1]), wall[1][0], wall[1][1],facecolor="dimgrey"))
#         #ax.add_patch(Circle(point[0], radius=point[1], facecolor='magenta'))
#         #ax.add_patch(Circle(goal[0], radius=goal[1], facecolor='darkorange'))

#         locations_x=[]
#         locations_y=[]
#         goals=[]
#         sampler=[]
#         for destination in locations[i]:

#             # if destination[1]>2e6:
#             #     continue

#             if  neighbour(destination[0:2],destination[-2:]):
#                 sample=destination[0:2]
#                 goals.append(sample)    
            
#             else:
#                 sample=destination[0:2]
#                 sampler.append(sample)
                
            
#             if len(sampler)==15:
#                 sample=random.choice(sampler)
#                 locations_x.append(sample[0])
#                 locations_y.append(sample[1])
#                 sampler=[]
        
#         print(i)
#         print(len(locations_x))
#         print(len(goals))
#         # samples=random.sample(samples,4000)


#         if len(goals)>=30:
#             goals=random.sample(goals,30)
        
#         for goal in goals:
#             locations_x.append(goal[0])
#             locations_y.append(goal[1])

#         print(len(locations_x))
#         print("*"*40)


#         # plot the locations
#         colors=[]
#         for k in range(len(locations_x)):
#             colors.append(k)
#         ax.scatter(locations_x,locations_y,marker='o',cmap='gist_rainbow',c=colors,s=16)
#         pcm = ax.get_children()[0]
#         plt.colorbar(pcm,ax=ax)
        
#         plt.xlim([-6,22])
#         plt.ylim([-6,22])
#         plt.xticks([])
#         plt.yticks([])
#         plt.savefig(address+"/agent"+str(i+1)+"/lcoations")

#     plt.close("all")    


def read(address,num_agents):

    success_rates=[]
    locations=[]
    explorations=[]

    for i in range(num_agents):
        with open(address+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
            success_rates.append(pickle.load(fp))
        
    for i in range(num_agents):
        with open(address+"/agent"+str(i+1)+"/locations", 'rb') as fp:
            locations.append(pickle.load(fp))
    
    # for i in range(num_agents):
    #     with open(address+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
    #         explorations.append(pickle.load(fp))
    
    
    return success_rates,locations,explorations




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--address',required=True)
    parser.add_argument('-n','--number',required=True)
    args = parser.parse_args()

    success_rates,locations,explorations=read(args.address,int(args.number))
    plot(args.address,success_rates,locations,explorations,int(args.number))















    