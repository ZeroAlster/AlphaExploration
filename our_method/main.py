import sys
sys.path.append("/home/futuhi/AlphaExploration")
import argparse
from general.maze import Env
from IPython.display import clear_output
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import pickle
import random
import statistics
from our_method.agent import Agent
from general.simple_estimator import minimum_visit
import torch
import os
import gym
import math
import mujoco_maze  # noqa
from general.simple_estimator import SEstimator
from wrappers import FetchWrapper
from general.FetchGraph import Model

# current version: Main method

#hyper params
######################################
max_frames  = 6e6
max_steps   = 100
batch_size  = 256
num_updates=30
checkpoints_interval=10000
evaluation_attempts=10
warm_up=20000
######################################

    

def evaluation(agent,environment):
    if environment=="maze":
        env_test=Env(n=max_steps,maze_type='square_large')
    elif environment=="point":
        env_test=gym.make("PointUMaze-v1")
    elif environment=="push":
        env_test=gym.make("PointPush-v1")
    elif environment=="fetch-slide":
        env_test=FetchWrapper(gym.make('FetchSlide-v1'))
    elif environment=="fetch-push":
        env_test=FetchWrapper(gym.make('FetchPush-v1'))
    elif environment=="fetch-reach":
        env_test=FetchWrapper(gym.make('FetchReach-v1'))
    else:
        raise TypeError("the environment is not allowed.")
    
    success=0
    for _ in range(evaluation_attempts):
        obs = env_test.reset()
        done=False
        while not done:
            action,_,_ = agent.get_action(obs,env=env_test,warmup=False,evaluation=True)
            obs,_,done,_= env_test.step(action[0])
        # if agent.neighbour(obs[0:2],obs[-2:]):
        if agent.neighbour(env_test.desired_goal,env_test.achieved_goal):
            success+=1
    
    return success/evaluation_attempts


def exploration(density):
    
    visits=np.copy(density.visits)-minimum_visit
    
    # define minimum visit to each cell
    if visits.shape[0]==20:
        min_val=1000
    elif visits.shape[0]==40:
        min_val=250
    elif visits.shape[0]==48:
        min_val=250
    elif visits.shape[0]==56:
        min_val=250
    else:
        sys.exit("wrong shape for density estimator")

    covered=(visits>=min_val).sum()
    all=visits.shape[0]*visits.shape[1]
    return covered/all


# main method to do updates
def save_to_buffer(agent,episode_memory,short=False):
   
    episode_next_state=episode_memory[-1][3]
    episode_done=episode_memory[-1][4]

    # This is a successful trajectory: MC update
    if episode_done:
        G=0
        for entry in reversed(episode_memory):
            G=entry[2]+agent.gamma*G
            if not short:
                agent.memory.push(entry[0],entry[1],G,entry[3],episode_done,1)
            else:
                agent.short_memory.push(entry[0],entry[1],G,entry[3],episode_done,1)    

    # This is an unsuccessful trajectory: longest n-step return
    else:
        i=0
        G=0
        for entry in reversed(episode_memory):
            G=entry[2]+agent.gamma*G
            agent.memory.push(entry[0],entry[1],G,episode_next_state,entry[4],i+1)
            i+=1


# This is used for one-step TD update
# def save_to_buffer(agent,episode_memory,short=False):
#     for entry in (episode_memory):
#         if not short:
#             agent.memory.push(entry[0],entry[1],entry[2],entry[3],entry[4],1)
#         else:
#             agent.short_memory.push(entry[0],entry[1],entry[2],entry[3],entry[4],1)


# This is used for average of first 8-step TD updates
# def save_to_buffer(agent,episode_memory,short=False):
#     for i in range(len(episode_memory)):
#         states=[]
#         rewards=[]
#         steps=[]
#         dones=[]
#         reward=0
#         step=0
#         for j in range(i,min(len(episode_memory),i+8)):
#             states.append(episode_memory[j][3])
#             dones.append(episode_memory[j][4])
#             reward+=math.pow(agent.gamma,step)*episode_memory[j][2]
#             step+=1
#             steps.append(step)
#             rewards.append(reward)


#         while len(states)<8:
#             states.append(episode_memory[i][0])
#             rewards.append(0)
#             steps.append(0)
#             dones.append(1)
                    
#         # append to memory
#         if not short:
#             agent.memory.push(episode_memory[i][0],episode_memory[i][1],rewards,states,dones,steps)
#         else:
#             agent.short_memory.push(episode_memory[i][0],episode_memory[i][1],rewards,states,dones,steps)
        


def train(agent,env,address,environment):
    
    #define variables for plotting
    success_rates=[]
    explorative_dist=[]
    success_num=0

    # fill all option length bins az zero
    for i in range(40):
        explorative_dist.append(0)

    # define required variables for maze environments
    if "fetch" not in environment:
        destinations=[]
        env_coverages=[]
        out_of_range=0
        density_height=env.observation_space.high[0]-env.observation_space.low[0]
        density_width=env.observation_space.high[1]-env.observation_space.low[1]
        env_density=SEstimator(0.5,density_height,density_width,[env.observation_space.low[0],env.observation_space.low[1]])
    
    # warmup period
    frame=0
    while frame<warm_up:
        state = env.reset()
        done=False
        episode_memory=[]
        terminal=state
        while not done:
            option,_,_ = agent.get_action(state,env=env,warmup=True)
            for action in option:
                next_state, reward, done, _ = env.step(action)
                
                if "fetch" not in environment:
                    episode_memory.append([state, action, reward, next_state, float(agent.neighbour(next_state[0:2],next_state[-2:]))])
                    # graph update if model is not available
                    if not agent.model_access:
                        agent.graph.add_transition(state,action,next_state)
                    # agent and env density update
                    agent.density_estimator.increment(state)
                    env_density.increment(state)
                    # recording the last location in each episode
                    terminal=state
                    # recording exploration rate during warmup
                    if frame % checkpoints_interval==0:
                        env_coverages.append(exploration(env_density))
                else:
                    episode_memory.append([state, action, reward, next_state, float(agent.neighbour(env.desired_goal,env.achieved_goal))])
                    agent.graph.add_transition(state,action,next_state)
                    agent.graph.increment(state)
                    if agent.neighbour(env.desired_goal,env.achieved_goal):
                        agent.graph.reset()
                
                state=next_state
                # check if episode is done
                frame+=1
                if done:
                    save_to_buffer(agent,episode_memory)
                    break
                    
        
        if "fetch" not in environment:
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
            option,e,l = agent.get_action(state,env=env,warmup=False)

            # record explorative option length
            if e:
                explorative_dist[l-1]+=1

            for action in option:
                next_state, reward, done,_= env.step(action)
                
                
                if "fetch" not in environment:
                    episode_memory.append([state, action, reward, next_state, float(agent.neighbour(next_state[0:2],next_state[-2:]))])
                    # graph update if model is not available
                    if not agent.model_access:
                        agent.graph.add_transition(state,action,next_state)
                    # agent and env density update
                    agent.density_estimator.increment(state)
                    env_density.increment(state)
                    # recording the last location in each episode
                    terminal=state
                else:
                    episode_memory.append([state, action, reward, next_state, float(agent.neighbour(env.desired_goal,env.achieved_goal))])
                    agent.graph.add_transition(state,action,next_state)
                    agent.graph.increment(state)
                    if agent.neighbour(env.desired_goal,env.achieved_goal):
                        agent.graph.reset()                    

                state=next_state
                
                # recording the success rates and exploration coverage after each checkpoint when the warmup is done
                frame+=1
                if frame % checkpoints_interval==0:
                    result=evaluation(agent,environment)
                    success_rates.append(result)
                    # env_coverages.append(exploration(env_density))
                    print("next checkpoint: "+str(frame)+"  steps")
                    print("goal is achieved: "+str(success_num))
                    print("success rate: "+str(result))
                    # print("out of range:"+str(out_of_range))
                    print("*"*50)
                        
                
                # adding successful trajectory to the short memory
                # if agent.neighbour(next_state[0:2],next_state[-2:])
                if agent.neighbour(env.desired_goal,env.achieved_goal):
                    success_num+=1
                    save_to_buffer(agent,episode_memory,short=True)
                
                # check if episode is done
                if done:
                    save_to_buffer(agent,episode_memory)
                    break

        if "fetch" not in environment:
            # recording terminal states
            destinations.append([terminal,frame])

        # set number of updates from short memory (off for one-buffer settings)
        agent.short_memory_updates=int((frame/max_frames)*num_updates)

        # update after each episode when the warmup is done
        for i in range(num_updates):
            agent.update(batch_size,i)
        

    # record training results: terminal states,success rates, environment coverage, and visits array
    # with open(address+"/locations", "wb") as fp:
    #         pickle.dump(destinations, fp)
    with open(address+"/success_rates", "wb") as fp:
            pickle.dump(success_rates, fp)
    # with open(address+"/env_coverage", "wb") as fp:
    #         pickle.dump(env_coverages, fp)
    with open(address+"/explorative_dist", "wb") as fp:
            pickle.dump(explorative_dist, fp)


def main(address,environment,model_avb):
    # initiate the environment, get action and state space size, and get action range
    if environment =="point":
        env=gym.make("PointUMaze-v1")
        num_actions=env.action_space.shape[0]
        num_states=env.observation_space.shape[0]
        action_range=np.array((1,0.25))
        density_estimator=SEstimator(1,20,20,[-2,-2])
        threshold=0.6
    elif environment=="maze":
        env=Env(n=max_steps,maze_type='square_large')
        num_actions = env.action_size
        num_states  = env.state_size*2
        action_range=np.array((env.action_range,env.action_range))
        density_estimator=SEstimator(1,10,10,[-0.5,-0.5])
        threshold=0.15
    elif environment=="push":
        env=gym.make("PointPush-v1")
        num_actions=env.action_space.shape[0]
        num_states=env.observation_space.shape[0]
        action_range=np.array((1,0.25))
        density_estimator=SEstimator(1,28,28,[-14,-2])
        threshold=0.6
    elif environment=="fetch-slide":
        env=FetchWrapper(gym.make('FetchSlide-v1'))
        num_actions=env.action_space.shape[0]
        num_states=env.observation_space["observation"].shape[0]+env.observation_space["desired_goal"].shape[0]
        threshold=0.05
        action_range=np.array((1,1,1,1))
        density_estimator=Model(env="slide")
    elif environment=="fetch-push":
        env=FetchWrapper(gym.make('FetchPush-v1'))
        num_actions=env.action_space.shape[0]
        num_states=env.observation_space["observation"].shape[0]+env.observation_space["desired_goal"].shape[0]
        threshold=0.05
        action_range=np.array((1,1,1,1))
        density_estimator=Model(env="push")
    elif environment=="fetch-reach":
        env=FetchWrapper(gym.make('FetchReach-v1'))
        num_actions=env.action_space.shape[0]
        num_states=env.observation_space["observation"].shape[0]+env.observation_space["desired_goal"].shape[0]
        threshold=0.05
        action_range=np.array((1,1,1,1))
        density_estimator=Model(env="reach")
    else:
        sys.exit("The environment does not exist!")

    
    # initiate the agent
    agent=Agent(num_actions,num_states,action_range,density_estimator,environment,threshold,model_avb)
    
    # train the agent
    train(agent,env,address,environment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--address',required=True)
    parser.add_argument('-e','--environment',required=True)
    parser.add_argument('-m','--model',required=True)
    args = parser.parse_args()

    
    # set random seeds
    seed=random.randint(0,1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    
    if  not os.path.exists(args.address):
        sys.exit("The path is wrong!")
    else:
        print("path is valid with seed: "+str(seed))
    
    # train the agent
    main(args.address,args.environment,args.model=="True")