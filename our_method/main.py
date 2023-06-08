import sys
sys.path.append("/nfs/home/futuhi/AlphaExploration")
import argparse
from general.maze import Env
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import numpy as np
import pickle
import random
import statistics
from our_method.agent import Agent
from general.simple_estimator import minimum_visit
import torch
import math
import os
from general.simple_estimator import SEstimator



# current version: one step TD for all samples


#hyper params
######################################
max_frames  = 6e6
max_steps   = 100
batch_size  = 128
num_updates=10
num_agents=9
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
            if destination[1]>4e6:
                continue

            sample=destination[0][0:2]
            
            if sample[0]>9.5 or sample[1]>9.5:
                #sample=[min(sample[0],9.5),min(sample[1],9.5)]
                continue
            if sample[0]<-0.5 or sample[1]<-0.5:
                #sample=[max(sample[0],-0.5),max(sample[1],-0.5)]
                continue

            if len(sampler)<4:
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


def exploration(density):
    min_val=1000
    visits=np.copy(density.visits)-minimum_visit
    covered=(visits>=min_val).sum()
    all=visits.shape[0]*visits.shape[1]
    return covered/all

# main method to do updates
# def save_to_buffer(agent,episode_memory,short=False):
   
#     episode_next_state=episode_memory[-1][3]
#     episode_done=episode_memory[-1][4]

#     # This is a successful trajectory: MC update
#     if episode_done:
#         G=0
#         for entry in reversed(episode_memory):
#             G=entry[2]+agent.gamma*G
#             if not short:
#                 agent.memory.push(entry[0],entry[1],G,entry[3],episode_done,1)
#             else:
#                 agent.short_memory.push(entry[0],entry[1],G,entry[3],episode_done,1)    
    

#     # This is an unsuccessful trajectory: longest n-step return
#     else:
#         i=0
#         G=0
#         for entry in reversed(episode_memory):
#             G=entry[2]+agent.gamma*G
#             agent.memory.push(entry[0],entry[1],G,episode_next_state,entry[4],i+1)
#             i+=1

# This is used for one-step TD update
# def save_to_buffer(agent,episode_memory,short=False):
#     for entry in (episode_memory):
#         if not short:
#             agent.memory.push(entry[0],entry[1],entry[2],entry[3],entry[4],1)
#         else:
#             agent.short_memory.push(entry[0],entry[1],entry[2],entry[3],entry[4],1)


# This is used for average of first 8-step TD updates
def save_to_buffer(agent,episode_memory,short=False):
    for i in range(len(episode_memory)):
        states=[]
        rewards=[]
        steps=[]
        dones=[]
        reward=0
        step=0
        for j in range(i,min(len(episode_memory),i+8)):
            states.append(episode_memory[j][3])
            dones.append(episode_memory[j][4])
            reward+=math.pow(agent.gamma,step)*episode_memory[j][2]
            step+=1
            steps.append(step)
            rewards.append(reward)
            
        
        # append to memory
        if not short:
            agent.memory.push(episode_memory[i][0],episode_memory[i][1],rewards,states,dones,steps)
        else:
            agent.short_memory.push(episode_memory[i][0],episode_memory[i][1],rewards,states,dones,steps)
        


def train(agent,env,address):
    
    #define variables for plotting
    destinations=[]
    success_rates=[]
    env_coverages=[]

    # define an estimator for the exploration curve
    env_density=SEstimator(0.5,10,10,[-0.5,-0.5])

    
    # warmup period
    frame=0
    while frame<warm_up:
        state = env.reset()
        done=False
        episode_memory=[]
        terminal=state
        while not done:
            option = agent.get_action(state,warmup=True)
            for action in option:
                next_state, reward, done, _ = env.step(action)
                episode_memory.append([state, action, reward, next_state, np.float(env.is_success)])
                state=next_state
                
                # agent and env density update
                agent.density_estimator.increment(state)
                env_density.increment(state)
                
                # recording the last location in each episode
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
            option = agent.get_action(state,warmup=False)
            for action in option:
                next_state, reward, done, _ = env.step(action)
                episode_memory.append([state, action, reward, next_state, np.float(env.is_success)])
                state=next_state

                # agent and env density update
                agent.density_estimator.increment(state)
                env_density.increment(state)
                
                # recording the last location in each episode
                terminal=state
                
                # recording the success rates and exploration coverage after each checkpoint when the warmup is done
                frame+=1
                if frame % checkpoints_interval==0:
                    print("next checkpoint: "+str(frame)+"  steps")
                    success_rates.append(evaluation(agent))
                    env_coverages.append(exploration(env_density))
                        
                
                # adding successful trajectory to the short memory
                if env.is_success:
                    save_to_buffer(agent,episode_memory,short=True)

                
                # check if episode is done
                if done:
                    save_to_buffer(agent,episode_memory)
                    break

        # recording terminal states
        destinations.append([terminal,frame])

        # set number of updates from short memory (off for one-buffer settings)
        agent.short_memory_updates=int((frame/max_frames)*num_updates)


        # update after each episode when the warmup is done
        for i in range(num_updates):
            agent.update(batch_size,i)
        

    # record training results: terminal states,success rates, environment coverage, and visits array
    with open(address+"/locations", "wb") as fp:
            pickle.dump(destinations, fp)
    with open(address+"/success_rates", "wb") as fp:
            pickle.dump(success_rates, fp)
    with open(address+"/successful_trajectories", "wb") as fp:
            pickle.dump(agent.short_memory.buffer, fp)
    with open(address+"/env_coverage", "wb") as fp:
            pickle.dump(env_coverages, fp)
    np.save(address+"/visits",agent.density_estimator.visits)

    # save agent nn models
    torch.save(agent.actor.state_dict(), address + '/actor.pth')
    torch.save(agent.critic.state_dict(), address + '/critic.pth')


    # print number of times that goal is chieved
    print("goal is achieved: "+str(env.goal_achievement)+"  times")



def main(address,environment):
    # initiate the environment, get action and state space size, and get action range
    if environment =="Pendulum":
        pass
    elif environment=="maze":
        env=Env(n=max_steps,maze_type='square_large')
        num_actions = env.action_size
        num_states  = env.state_size*2
        action_range=env.action_range
    else:
        sys.exit("The environment does not exist!")

    
    # initiate the agent
    agent=Agent(num_actions,num_states,action_range)
    
    # train the agent
    train(agent,env,address)


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