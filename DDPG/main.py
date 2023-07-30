import sys
sys.path.append("/nfs/home/futuhi/AlphaExploration")
import argparse
from DDPG.ddpg import Agent
from general.maze import Env
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import statistics
from general.simple_estimator import minimum_visit
import torch
import os
from general.simple_estimator import SEstimator
import gym
import mujoco_maze #noqa


#hyper params
######################################
max_frames  = 6e6
max_steps   = 100
batch_size  = 128
num_updates=10
num_agents=10
checkpoints_interval=10000
evaluation_attempts=10
warm_up=20000
######################################



def plot(address,locations,success_rates,explorations):

    # plot success rates
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
    plt.plot(horizon[0,:],mean[0,:], 'k-',color="blue")

    # fix the error bar
    std=std
    down_bar=np.maximum((mean-std)[0,:],0)
    up_bar=np.minimum((mean+std)[0,:],1)

    plt.fill_between(horizon[0,:],down_bar,up_bar)
    plt.savefig(address+"/success_rates.png")


    #plot the exploration curves
    number=100
    std=np.zeros((1,number))
    mean=np.zeros((1,number))
    horizon=np.zeros((1,number))
    
    
    for i in range(number):
        values=[]
        for j in range(num_agents):
            values.append(explorations[j][i])
        mean[0][i]=sum(values)/len(values)
        std[0][i]=statistics.pstdev(values)
        horizon[0][i]=i
    
    plt.figure()
    plt.plot(horizon[0,:],mean[0,:], 'k-',color="blue")

    # fix the error bar
    down_bar=np.maximum((mean-std)[0,:],0)
    up_bar=np.minimum((mean+std)[0,:],1)

    plt.fill_between(horizon[0,:],down_bar,up_bar)
    plt.savefig(address+"/env_coverage.png")

    
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

            # we plot the results until 6M frames
            if destination[1]>6e6:
                continue

            sample=destination[0][0:2]
            
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
    

def evaluation(agent,environment):
    if environment=="maze":
        env_test=Env(n=max_steps,maze_type='square_large')
    else:
        env_test=gym.make("PointUMaze-v1")
    
    success=0
    for _ in range(evaluation_attempts):
        obs = env_test.reset()
        done=False
        while not done:
            action = agent.get_action(obs,evaluation=True)[0]
            obs,_, done,_= env_test.step(action)
        if agent.neighbour(obs[0:2],obs[-2:]):
            success+=1
    
    return success/evaluation_attempts


def exploration(density):
    
    visits=np.copy(density.visits)-minimum_visit
    
    # define minimum visit to each cell
    if visits.shape[0]==20:
        min_val=1000
    elif visits.shape[0]==40:
        min_val=250
    else:
        sys.exit("wrong shape for density estimator")
    
    covered=(visits>=min_val).sum()
    all=visits.shape[0]*visits.shape[1]
    return covered/all


def save_to_buffer(agent,episode_memory):
        
    # this is the same for all trajectories: one-step TD
    for entry in reversed(episode_memory):
        agent.memory.push(entry[0],entry[1],entry[2],entry[3],entry[4],1)


def train(agent,env,address,environment):
    
    #define variables for plotting
    destinations=[]
    success_rates=[]
    env_coverages=[]

    # define an estimator for the exploration curve
    density_height=env.observation_space.high[0]-env.observation_space.low[0]
    env_density=SEstimator(0.5,density_height,density_height,[env.observation_space.low[0],env.observation_space.low[1]])

    
    # warmup period
    frame=0
    while frame<warm_up:
        state = env.reset()
        done=False
        episode_memory=[]
        terminal=state
        while not done:
            option = agent.get_action(state)
            for action in option:
                next_state, reward, done, _ = env.step(action)
                episode_memory.append([state, action, reward, next_state, np.float(agent.neighbour(next_state[0:2],next_state[-2:]))])
                state=next_state
                
                # env density update
                env_density.increment(state)
                
                # recording the last loation in each episode
                terminal=state
                
                
                # recording exploration rate during warmup
                if frame % checkpoints_interval==0:
                    env_coverages.append(exploration(env_density))
                
                # check if episode is done
                frame+=1
                if done:
                    save_to_buffer(agent,episode_memory)
                    break
                    
        
        # recording terminal states
        destinations.append([terminal,frame])


    print("warmup has ended!")
    
    
    # train and save the model
    while frame<max_frames:
        
        state = env.reset()
        done=False
        episode_memory=[]
        terminal=state
        
        while not done:
            option = agent.get_action(state)
            for action in option:
                next_state, reward, done, _ = env.step(action)
                episode_memory.append([state, action, reward, next_state, np.float(agent.neighbour(next_state[0:2],next_state[-2:]))])
                state=next_state

                # env density update
                env_density.increment(state)
                
                # recording the last loation in each episode
                terminal=state
                
                # recording the success rates and exploration coverage after each checkpoint when the warmup is done
                frame+=1
                if frame % checkpoints_interval==0:
                    print("next checkpoint: "+str(frame)+"  steps")
                    success_rates.append(evaluation(agent,environment))
                    env_coverages.append(exploration(env_density))
                
                # check if episode is done
                if done:
                    save_to_buffer(agent,episode_memory)
                    break

        # recording terminal states
        destinations.append([terminal,frame])

        # update after each episode when the warmup is done
        for i in range(num_updates):
            agent.update(batch_size)
        

    # record training results: terminal states,success rates, environment coverage, and visits array
    with open(address+"/locations", "wb") as fp:
            pickle.dump(destinations, fp)
    with open(address+"/success_rates", "wb") as fp:
            pickle.dump(success_rates, fp)
    with open(address+"/env_coverage", "wb") as fp:
            pickle.dump(env_coverages, fp)

    # save agent nn models
    torch.save(agent.actor.state_dict(), address + '/actor.pth')
    torch.save(agent.critic.state_dict(), address + '/critic.pth')


    # print number of times that goal is chieved
    print("goal is achieved: "+str(env.goal_achievement)+"  times")



def main(address,environment):
    # initiate the environment, get action and state space size, and get action range
    if environment =="point":
        env=gym.make("PointUMaze-v1")
        num_actions=env.action_space.shape[0]
        num_states=env.observation_space.shape[0]
        action_range=np.array((1,0.25))
        threshold=0.6
    elif environment=="maze":
        env=Env(n=max_steps,maze_type='square_large')
        num_actions = env.action_size
        num_states  = env.state_size*2
        action_range=np.array((env.action_range,env.action_range))
        threshold=0.15
    else:
        sys.exit("The environment does not exist!")

    
    # initiate the agent
    agent=Agent(num_actions,num_states,action_range,threshold)
    
    # train the agent
    train(agent,env,address,environment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--address',required=True)
    parser.add_argument('-t','--task',required=True)
    parser.add_argument('-e','--environment',required=True)
    args = parser.parse_args()

    
    # set random seeds
    seed=random.randint(0,100)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    
    if  not os.path.exists(args.address):
        sys.exit("The path is wrong!")
    else:
        print("path is valid!")
    
    if args.task=="train":
        main(args.address,args.environment)
    elif args.task=="plot":
        # plot success rates, exploration curves, and destinations of all agents
        success_rates=[]
        locations=[]
        explorations=[]
        
        for i in range(num_agents):
            with open(args.address+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
                success_rates.append(pickle.load(fp))
        
        for i in range(num_agents):
            with open(args.address+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
                explorations.append(pickle.load(fp))
        
        for i in range(num_agents):
            with open(args.address+"/agent"+str(i+1)+"/locations", 'rb') as fp:
                destinations=pickle.load(fp)
            locations.append([destinations,args.address+"/agent"+str(i+1)])
        
        plot(args.address,success_rates=success_rates,locations=locations,explorations=explorations)

    else:
        pass