import argparse
from env import Env
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import numpy as np
import pickle
import random
import statistics
from agent import Agent
from simple_estimator import minimum_visit
import torch
import math
import sys
import os


#hyper params
######################################
max_frames  = 8e6
max_steps   = 50
batch_size  = 128
num_updates=10
num_agents=2
checkpoints_interval=10000
evaluation_attempts=5
warm_up=20000
stored_points_for_cell=10
stored_trajectories_length=40
alpha=0.5
alpha_decay=0.9999997
######################################


# cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# set random seeds
torch.manual_seed(10)
torch.cuda.manual_seed(10)
random.seed(10)
np.random.seed(10)


def plot(address,locations,success_rates):

    # plot success rates
    std=np.zeros((1,len(success_rates[0])))
    mean=np.zeros((1,len(success_rates[0])))
    horizon=np.zeros((1,len(success_rates[0])))
    for i in range(len(success_rates[0])):
        values=[]
        for j in range(num_agents):
            values.append(success_rates[j][i])
        mean[0][i]=sum(values)/len(values)
        std[0][i]=statistics.pstdev(values)
        horizon[0][i]=i
    
    plt.plot(horizon[0,:],mean[0,:], 'k-',color="blue")
    plt.fill_between(horizon[0,:],(mean-std)[0,:], (mean+std)[0,:])
    plt.savefig(address+"/success_rates.png")

    
    # plot locations on the map
    for location_address in locations:
        
        env=Env(n=max_steps,maze_type='square_large')
        destinations=location_address[0]
        
        # plot the map first
        _, ax = plt.subplots(1, 1, figsize=(5, 4))
        for x, y in env.maze._walls:
            ax.plot(x, y, 'k-')
        
        # remove the outliers
        locations_x=[]
        locations_y=[]
        sampler=[]
        for destination in destinations:
            sample=destination[0:2]
            
            if sample[0]>9.5 or sample[1]>9.5:
                #sample=[min(sample[0],9.5),min(sample[1],9.5)]
                continue
            if sample[0]<-0.5 or sample[1]<-0.5:
                #sample=[max(sample[0],-0.5),max(sample[1],-0.5)]
                continue

            if len(sampler)<2:
                sampler.append(sample)
            else:
                sample=random.choice(sampler)
                locations_x.append(sample[0])
                locations_y.append(sample[1])
                sampler=[]
        
        # plot the locations
        colors=[]
        for i in range(len(locations_x)):
            colors.append(i)
        ax.scatter(locations_x,locations_y,marker='o',cmap='gist_rainbow',c=colors,s=16)
        pcm = ax.get_children()[0]
        plt.colorbar(pcm,ax=ax)
        plt.savefig(location_address[1]+"/destinations.png")

    plt.close("all")



def point_to_cell(coordination,density_estimator):
    cell_x=math.floor((coordination[0]-density_estimator.env_start[0])/density_estimator.cell_side)
    cell_y=math.floor((coordination[1]-density_estimator.env_start[1])/density_estimator.cell_side)

    cell_x=min(cell_x,density_estimator.visits.shape[0]-1)
    cell_x=max(cell_x,0)
    cell_y=min(cell_y,density_estimator.visits.shape[1]-1)
    cell_y=max(cell_y,0)

    return cell_x,cell_y


def start_from_memory(agent,env,state):
    episode_memory=[]
    done=False
    trajectory=[]
    visits=np.copy(agent.density_estimator.visits)-minimum_visit
    
    position=(0,0)
    for i in range(visits.shape[0]):
        for j in range(visits.shape[1]):
            if visits[i][j]>0 and len(agent.path_array[i][j])>0:
                if visits[position[0]][position[1]]>visits[i][j]:
                    position=(i,j)

    cell_x=position[0]
    cell_y=position[1]
    
    
    path=random.choice(agent.path_array[cell_x][cell_y])
    for action in path:
        next_state, reward, done, _ = env.step(action)
        episode_memory.append([state, action, reward, next_state, np.float(env.is_success)])
        state=next_state
        agent.density_estimator.increment(state)
        trajectory.append(action)
    
    
    return trajectory,state,done,episode_memory
    


def evaluation(agent):
    env_test=Env(n=max_steps,maze_type='square_large')
    success=0
    for _ in range(evaluation_attempts):
        obs = env_test.reset()
        done=False
        while not done:
            action = agent.get_action(obs,warmup=False,evaluation=True)[0]
            obs,_, done,_= env_test.step(action)
        if env_test.is_success:
            success+=1
    
    return success/evaluation_attempts



# need to be analyzed again.
def save_to_memory(agent,episode_memory,short=False):

    episode_reward=episode_memory[-1][2]
    episode_next_state=episode_memory[-1][3]
    episode_done=episode_memory[-1][4]

    # this is a successful trajectory: MC update
    if episode_done:
        i=0
        for entry in reversed(episode_memory):
            reward=episode_reward*math.pow(agent.gamma,i)
            if not short:
                agent.memory.push(entry[0],entry[1],reward,episode_next_state,episode_done,i+1)
            else:
                agent.short_memory.push(entry[0],entry[1],reward,episode_next_state,episode_done,i+1)
            i+=1    
    
    # this is an unsuccessful trajectory: truncated MC update
    else:
        i=0
        for entry in reversed(episode_memory):
            reward=episode_reward*math.pow(agent.gamma,i)
            agent.memory.push(entry[0],entry[1],reward,episode_next_state,episode_done,i+1)
            i+=1


def train(agent,env,address):
    
    #define variables for plotting
    destinations=[]
    success_rates=[]
    global alpha
    
    
    # warmup period
    frame=0
    while frame<warm_up:
        state = env.reset()
        done=False
        trajectory=[]
        episode_memory=[]
        terminal=state
        while not done:
            option = agent.get_action(state,warmup=True)
            for action in option:
                next_state, reward, done, _ = env.step(action)
                episode_memory.append([state, action, reward, next_state, np.float(env.is_success)])
                state=next_state
                agent.density_estimator.increment(state)
                terminal=state
                
                # storing trajectories
                if len(trajectory)<stored_trajectories_length:
                    trajectory.append(action)
                    cell_x,cell_y=point_to_cell(state,agent.density_estimator)
                    if len(agent.path_array[cell_x][cell_y]) < stored_points_for_cell:
                        agent.path_array[cell_x][cell_y].append(trajectory.copy())
                
                # check if episode is done
                frame+=1
                if done:
                    save_to_memory(agent,episode_memory)
                    break
                    
        
        # recording terminal states
        destinations.append(terminal)


    print("warmup has ended!")
    
    
    # train and save the model
    while frame<max_frames:
        state = env.reset()
        
        # check if agent wants to use its memory
        if random.uniform(0, 1)<alpha:
            trajectory,state,done,episode_memory=start_from_memory(agent,env,state)
        else:
            done=False
            trajectory=[]
            episode_memory=[]

        terminal=state
        
        while not done:
            option = agent.get_action(state,warmup=False)
            for action in option:
                next_state, reward, done, _ = env.step(action)
                episode_memory.append([state, action, reward, next_state, np.float(env.is_success)])
                state=next_state
                agent.density_estimator.increment(state)
                terminal=state
                
                # storing trajectories
                if len(trajectory)<stored_trajectories_length:
                    trajectory.append(action)
                    cell_x,cell_y=point_to_cell(state,agent.density_estimator)
                    if len(agent.path_array[cell_x][cell_y]) < stored_points_for_cell:
                        agent.path_array[cell_x][cell_y].append(trajectory.copy())
                
                # recording the success rates after each checkpoint when the warmup is done
                frame+=1
                if frame % checkpoints_interval==0:
                        print("next checkpoint: "+str(frame)+"  steps")
                        success_rates.append(evaluation(agent))
                
                # adding successful trajectory to the short memory
                if env.is_success:
                    save_to_memory(agent,episode_memory,short=True)
                
                # check if episode is done
                if done:
                    save_to_memory(agent,episode_memory)
                    break
                
                # decaying the alpha
                alpha=alpha*alpha_decay

        # recording terminal states
        destinations.append(terminal)

        # update number of updates from short memory
        if frame>int(2e6):
            agent.short_memory_updates=int((frame/max_frames)*num_updates)

        # update after each episode when the warmup is done
        for i in range(num_updates):
            agent.update(batch_size,i)
        

    # record training results: terminal states,success rates, and visits array
    with open(address+"/locations", "wb") as fp:
            pickle.dump(destinations, fp)
    with open(address+"/success_rates", "wb") as fp:
            pickle.dump(success_rates, fp)
    with open(address+"/successful_trajectories", "wb") as fp:
            pickle.dump(agent.short_memory.buffer, fp)
    np.save(address+"/visits",agent.density_estimator.visits)

    # save agent nn models
    torch.save(agent.actor.state_dict(), address + '/actor.pth')
    torch.save(agent.critic.state_dict(), address + '/critic.pth')


    # print number of times that goal is chieved
    print("goal is achieved: "+str(env.goal_achievement)+"  times")



def main(address):
    # initiate the environment, get action and state space size, and get action range
    env=Env(n=max_steps,maze_type='square_large')
    num_actions = env.action_size
    num_states  = env.state_size*2
    action_range=env.action_range
    
    # initiate the agent
    agent=Agent(num_actions,num_states,action_range)
    
    # train the agent
    train(agent,env,address)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--address',required=True)
    parser.add_argument('-t','--task',required=True)
    args = parser.parse_args()
    
    if  not os.path.exists(args.address):
        sys.exit("The path is wrong!")
    else:
        print("path is valid!")
    
    if args.task=="train":
        main(args.address)
    elif args.task=="plot":
        # plot success rates and destinations of all agents
        success_rates=[]
        locations=[]
        
        for i in range(num_agents):
            with open(args.address+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
                success_rates.append(pickle.load(fp))
        
        for i in range(num_agents):
            with open(args.address+"/agent"+str(i+1)+"/locations", 'rb') as fp:
                destinations=pickle.load(fp)
            locations.append([destinations,args.address+"/agent"+str(i+1)])
        
        plot(args.address,success_rates=success_rates,locations=locations)

    else:
        pass