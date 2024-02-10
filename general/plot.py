import pickle
import sys
sys.path.append("/home/futuhi/AlphaExploration")
import math
import matplotlib.pyplot as plt
import numpy as np
import statistics
import matplotlib
import random
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import argparse
from general.maze import Env



def neighbour(state,goal):
        if math.sqrt(math.pow(state[0]-goal[0],2)+math.pow(state[1]-goal[1],2))<=0.6:
            return True
        else:
            return False
        

def plot(address,success_rates,locations,explorations,num_agents):
    
    # define structure of the maze
    walls=[[(-18,-6),(36,4)],[(-18,-6),(4,36)],[(-18,26),(36,4)],[(14,-6),(4,36)]]
    # ,[(-18,6),(8,4)],[(-6,6),(16,4)],[(6,18),(4,-12)],[(6,6),(8,4)],[(14,2),(-8,-4)],[(2,-2),(-20,4)]]
    # point=[(0,0),1]
    goal=[(4,24.8),0.6]   
    
    # extra blocks
    walls.append([(-18,6),(8,4)])
    walls.append([(-6,6),(16,4)])
    walls.append([(6,10),(4,8)])
    walls.append([(6,18),(8,-4)])
    walls.append([(18,26),(-12,-4)])
    walls.append([(2,26),(-16,-4)])
    
    
    for i in range(len(locations)):

        # plot the maze and goal first
        _, ax = plt.subplots()
        
        locations_x=[]
        locations_y=[]
        goals=[]
        sampler=[]
        for destination in locations[i]:

            # if destination[1]>4e6:
            #     continue

            # destination=destination[0]
            if  neighbour(destination[0:2],destination[-2:]):
                sample=destination[0:2]
                goals.append(sample)
            
            else:
                sample=destination[0:2]
                sampler.append(sample)
                
            
            if len(sampler)==2:
                sample=random.choice(sampler)
                locations_x.append(sample[0])
                locations_y.append(sample[1])
                sampler=[]
        
        print(i)
        print(len(locations_x))
        print(len(goals))
        
        # cut to the spesific length
        # maximum=2200
        # if len(locations_x)>maximum:
        #     locations_x=locations_x[0:maximum]
        #     locations_y=locations_y[0:maximum]


        if len(goals)>=30:
            goals=random.sample(goals,min(250,len(goals)-1))
        
        for goal in goals:
            locations_x.append(goal[0])
            locations_y.append(goal[1])

        print(len(locations_x))
        print("*"*40)


        # plot the locations
        colors=[]
        for k in range(len(locations_x)):
            colors.append(k)
        ax.scatter(locations_x,locations_y,marker='o',cmap='gist_rainbow',c=colors,s=16)
        
        ax=plt.gca() #get the current axes
        for pcm in ax.get_children():
            if isinstance(pcm, matplotlib.cm.ScalarMappable):
                break
        plt.colorbar(pcm,ax=ax)


        #plot the walls
        for wall in walls:
            ax.add_patch(Rectangle((wall[0][0], wall[0][1]), wall[1][0], wall[1][1],facecolor="dimgrey"))
        
        plt.xlim([-18,18])
        plt.ylim([-6,30])
        plt.xticks([])
        plt.yticks([])
        plt.savefig(address+"/agent"+str(i+1)+"/lcoations")

    plt.close("all")    


        

# def plot(address,success_rates,locations,explorations,num_agents):
    
    # plot locations on the map
#     t=1
#     for location_address in locations:
        
#         env=Env(n=max_steps,maze_type='square_large')
#         destinations=location_address

#         # plot the map first
#         _, ax = plt.subplots(1, 1, figsize=(5, 4))
#         for x, y in env.maze._walls:
#             ax.plot(x, y, 'k-')
        
#         # remove the outliers
#         locations_x=[]
#         locations_y=[]
#         sampler=[]

        
#         for destination in destinations:

#             # we plot the results until 6M frames
#             # if destination[1]>4e6:
#             #     continue
#             sample=destination[0][0:2]
            
#             if sample[0]>9.5 or sample[1]>9.5:
#                 #sample=[min(sample[0],9.5),min(sample[1],9.5)]
#                 continue
#             if sample[0]<-0.5 or sample[1]<-0.5:
#                 #sample=[max(sample[0],-0.5),max(sample[1],-0.5)]
#                 continue

#             if len(sampler)<5:
#                 sampler.append(sample)
#             else:
#                 sample=random.choice(sampler)
#                 locations_x.append(sample[0])
#                 locations_y.append(sample[1])
#                 sampler=[]
        
#         # plot the locations
#         colors=[]
#         for i in range(len(locations_x)):
#             colors.append(i)
#         ax.scatter(locations_x,locations_y,marker='o',cmap='gist_rainbow',c=colors,s=16)
#         # pcm = ax.get_children()[0]

#         ax=plt.gca() #get the current axes
#         for pcm in ax.get_children():
#             if isinstance(pcm, matplotlib.cm.ScalarMappable):
#                 break
#         plt.colorbar(pcm,ax=ax)
#         plt.savefig(address+"/agent"+str(t)+"/destinations.png")
#         t+=1

#     plt.close("all")


# def plot(address,success_rates,locations,explorations,num_agents):
        
    # define structure of the maze
    # walls=[[(-6,-6),(4,28)],[(-6,-6),(28,4)],[(-6,18),(28,4)],[(18,-6),(4,28)],[(-6,2),(16,8)]]
    # point=[(0,0),1]
    # goal=[(0,16),0.6]

    # # extra blocks
    # walls.append([(2,14),(4,4)])
    # walls.append([(14,10),(4,4)])    
    
    # for i in range(len(locations)):

    #     # plot the maze and goal first
    #     _, ax = plt.subplots()
    #     for wall in walls:
    #         ax.add_patch(Rectangle((wall[0][0], wall[0][1]), wall[1][0], wall[1][1],facecolor="dimgrey"))
    #     #ax.add_patch(Circle(point[0], radius=point[1], facecolor='magenta'))
    #     #ax.add_patch(Circle(goal[0], radius=goal[1], facecolor='darkorange'))

    #     locations_x=[]
    #     locations_y=[]
    #     goals=[]
    #     sampler=[]
    #     for destination in locations[i]:

    #         # if destination[1]>2e6:
    #         #     continue

    #         # destination=destination[0]
    #         if  neighbour(destination[0:2],destination[-2:]):
    #             sample=destination[0:2]
    #             goals.append(sample)    
            
    #         else:
    #             sample=destination[0:2]
    #             sampler.append(sample)
                
            
    #         if len(sampler)==50:
    #             sample=random.choice(sampler)
    #             locations_x.append(sample[0])
    #             locations_y.append(sample[1])
    #             sampler=[]
        
    #     print(i)
    #     print(len(locations_x))
    #     print(len(goals))
    #     # samples=random.sample(samples,4000)


    #     if len(goals)>=30:
    #         goals=random.sample(goals,10)
        
    #     for goal in goals:
    #         locations_x.append(goal[0])
    #         locations_y.append(goal[1])

    #     print(len(locations_x))
    #     print("*"*40)


    #     # plot the locations
    #     colors=[]
    #     for k in range(len(locations_x)):
    #         colors.append(k)
    #     ax.scatter(locations_x,locations_y,marker='o',cmap='gist_rainbow',c=colors,s=16)
        
    #     ax=plt.gca() #get the current axes
    #     for pcm in ax.get_children():
    #         if isinstance(pcm, matplotlib.cm.ScalarMappable):
    #             break
    #     plt.colorbar(pcm,ax=ax)
        
    #     plt.xlim([-6,22])
    #     plt.ylim([-6,22])
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.savefig(address+"/agent"+str(i+1)+"/lcoations")

    # plt.close("all")    


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















    