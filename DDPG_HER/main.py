import sys
sys.path.append("/home/futuhi/AlphaExploration")
import argparse
from DDPG_HER.ddpg_her import Agent
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
num_updates=20
num_agents=10
checkpoints_interval=10000
evaluation_attempts=10
warm_up=20000
######################################
   

def evaluation(agent,environment):
    if environment=="maze":
        env_test=Env(n=max_steps,maze_type='square_large')
    elif environment=="push":
        env_test=gym.make("PointPush-v1")
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
    elif visits.shape[0]==56:
        min_val=250
    else:
        sys.exit("wrong shape for density estimator")
    
    covered=(visits>=min_val).sum()
    all=visits.shape[0]*visits.shape[1]
    return covered/all


def save_to_buffer(agent,episode_memory,k,HER_goal_reward):
        
    # this is the same for all trajectories: one-step TD
    for t in range(len(episode_memory)):
        agent.memory.push(episode_memory[t][0],episode_memory[t][1],episode_memory[t][2],episode_memory[t][3],episode_memory[t][4],1)


        # This is for HER
        if t<len(episode_memory)-1:
            for _ in range(k):
                index=random.randint(t+1,len(episode_memory)-1)
                goal=episode_memory[index][0][0:2]
                
                # copy former transition
                transition=[]
                transition.append(episode_memory[t][0].copy())
                transition.append(episode_memory[t][1].copy())
                transition.append(1)
                transition.append(episode_memory[t][3].copy())
                transition.append(1)
                
                
                transition[0][-2:]=goal
                transition[3][-2:]=goal
                if agent.neighbour(transition[3][0:2],transition[3][-2:]):
                    transition[2]=HER_goal_reward
                    transition[4]=1
                else:
                    transition[2]=episode_memory[0][2]
                    transition[4]=0
                
                agent.memory.push(transition[0],transition[1],transition[2],transition[3],transition[4],1)


def train(agent,env,address,environment,k,her_goal):
    
    #define variables for plotting
    destinations=[]
    success_rates=[]
    env_coverages=[]
    success_num=0
    out_of_range=0

    # define an estimator for the exploration curve
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
            option = agent.get_action(state)
            for action in option:
                next_state, reward, done, _ = env.step(action)
                episode_memory.append([state, action, reward, next_state, float(agent.neighbour(next_state[0:2],next_state[-2:]))])
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
                    save_to_buffer(agent,episode_memory,k,her_goal)
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
                episode_memory.append([state, action, reward, next_state, float(agent.neighbour(next_state[0:2],next_state[-2:]))])
                state=next_state

                # skip the states out of range
                if state[1]>env.observation_space.high[1] or state[1]<env.observation_space.low[1] or state[0]<env.observation_space.low[0] or state[0]>env.observation_space.high[0]:
                    out_of_range+=1
                    break

                # env density update
                env_density.increment(state)
                
                # recording the last loation in each episode
                terminal=state

                # record number of goal ahievements
                if agent.neighbour(next_state[0:2],next_state[-2:]):
                    success_num+=1
                
                # recording the success rates and exploration coverage after each checkpoint when the warmup is done
                frame+=1
                if frame % checkpoints_interval==0:
                    success_rates.append(evaluation(agent,environment))
                    env_coverages.append(exploration(env_density))
                    print("next checkpoint: "+str(frame)+"  steps")
                    print("out of range: "+str(out_of_range))
                    print("goal is achieved: "+str(success_num))
                
                # check if episode is done
                if done:
                    save_to_buffer(agent,episode_memory,k,her_goal)
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



def main(address,environment,k):
    # initiate the environment, get action and state space size, and get action range
    if environment =="point":
        env=gym.make("PointUMaze-v1")
        num_actions=env.action_space.shape[0]
        num_states=env.observation_space.shape[0]
        action_range=np.array((1,0.25))
        threshold=0.6
        HER_goal_reward=1
    elif environment=="maze":
        env=Env(n=max_steps,maze_type='square_large')
        num_actions = env.action_size
        num_states  = env.state_size*2
        action_range=np.array((env.action_range,env.action_range))
        threshold=0.15
        HER_goal_reward=10
    elif environment=="push":
        env=gym.make("PointPush-v1")
        num_actions=env.action_space.shape[0]
        num_states=env.observation_space.shape[0]
        action_range=np.array((1,0.25))
        threshold=0.6
        HER_goal_reward=1
    else:
        sys.exit("The environment does not exist!")

    
    # initiate the agent
    agent=Agent(num_actions,num_states,action_range,threshold)
    
    # train the agent
    train(agent,env,address,environment,k,HER_goal_reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--address',required=True)
    parser.add_argument('-e','--environment',required=True)
    parser.add_argument('-k','--number_of_goals',required=True)
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
        print("path is valid!")
    
    main(args.address,args.environment,int(args.number_of_goals))