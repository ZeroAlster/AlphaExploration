import sys
sys.path.append("/home/futuhi/AlphaExploration")
import argparse
from general.maze import Env
import numpy as np
import pickle
import random
from our_method.agent import Agent
from general.simple_estimator import minimum_visit
import torch
import copy
import os
import gym
import mujoco_maze  # noqa
import imageio
from general.simple_estimator import SEstimator
from wrappers import FetchWrapper
from general.MetaGraph import Model
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
import matplotlib.pyplot as plt


# current version: Main method

#hyper params
######################################
max_frames  = 2e6
max_steps   = 100
batch_size  = 512
num_updates=250
checkpoints_interval=10000
evaluation_attempts=10
warm_up=500
######################################


def evaluation(agent,env):
    
    success=0
    for _ in range(evaluation_attempts):
        obs = env.reset()
        done=False
        while (not done) and (not env.success):
            action,_,_ = agent.get_action(obs,warmup=False,evaluation=True)
            obs,_,done,_= env.step(action[0])
        
        if env.success:
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
        


def train(agent,env,address,environment,test_env):
    
    print('Setting environment variable to use GPU rendering:')
    os.environ["MUJOCO_GL"]="egl"
    
    #define variables for plotting
    success_rates=[]
    explorative_dist=[]
    success_num=0
    destinations=[]

    # fill all option length bins az zero
    for _ in range(40):
        explorative_dist.append(0)

    if "v2" not in environment:
        env_coverages=[]
        out_of_range=0
        density_height=env.observation_space.high[0]-env.observation_space.low[0]
        density_width=env.observation_space.high[1]-env.observation_space.low[1]
        env_density=SEstimator(0.5,density_height,density_width,[env.observation_space.low[0],env.observation_space.low[1]])
    
    # warmup period
    frame=0
    while frame<warm_up:
        
        episode_memory=[]
        state= env.reset()
        done=False
        terminal=state
        
        while (not done) and (not env.success):
            option,_,_ = agent.get_action(state,warmup=True)
            for action in option:
                next_state, reward, done, _ = env.step(action)

                if "v2" not in environment:
                    episode_memory.append([state, action, reward, next_state, float(agent.neighbour(next_state[0:2],next_state[-2:]))])
                    # graph update if model is not available
                    if not agent.model_access:
                        agent.graph.add_transition(state,action,next_state)
                    # agent and env density update
                    agent.density_estimator.increment(state)
                    env_density.increment(state)
                    # recording exploration rate during warmup
                    if frame % checkpoints_interval==0:
                        env_coverages.append(exploration(env_density))
                else:
                    episode_memory.append([state, action, reward, next_state, float(env.success)])
                    # agent.graph.add_transition(state,action,next_state)
                    agent.graph.increment(state)
                
                state=next_state
                frame+=1

                # recording the last location in each episode
                terminal=state

                # check if episode is done
                if env.success:
                    success_num+=1
                    save_to_buffer(agent,episode_memory,short=True)
                    save_to_buffer(agent,episode_memory)
                    break
    
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

        while (not done) and (not env.success):
            option,e,l = agent.get_action(state,warmup=False,data=env.data)            
            
            # record explorative option length
            if e:
                explorative_dist[l-1]+=1

            for action in option:
            
                next_state, reward, done,_= env.step(action)


                if "v2" not in environment:
                    episode_memory.append([state, action, reward, next_state, float(agent.neighbour(next_state[0:2],next_state[-2:]))])
                    # graph update if model is not available
                    if not agent.model_access:
                        agent.graph.add_transition(state,action,next_state)
                    # agent and env density update
                    agent.density_estimator.increment(state)
                    env_density.increment(state)
                else:
                    episode_memory.append([state, action, reward, next_state, float(env.success)])
                    # agent.graph.add_transition(state,action,next_state) 
                    agent.graph.increment(state)                   

                state=next_state
                frame+=1

                # recording the last location in each episode
                terminal=state

                # recording the success rates and exploration coverage after each checkpoint when the warmup is done
                if frame % checkpoints_interval==0:
                    result=evaluation(agent,test_env)
                    success_rates.append(result)
                    # env_coverages.append(exploration(env_density))
                    print("next checkpoint: "+str(frame)+"  steps")
                    print("goal is achieved: "+str(success_num))
                    print("success rate: "+str(result))
                    # print("out of range:"+str(out_of_range))
                    print("*"*50)
                        
                
                # adding successful trajectory to the short memory
                # if agent.neighbour(next_state[0:2],next_state[-2:])
                if env.success:
                    success_num+=1
                    save_to_buffer(agent,episode_memory,short=True)
                    save_to_buffer(agent,episode_memory)
                    break
                
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
    # with open(address+"/env_coverage", "wb") as fp:
    #         pickle.dump(env_coverages, fp)
    with open(address+"/explorative_dist", "wb") as fp:
            pickle.dump(explorative_dist, fp)


def main(address,environment,model_avb,seed):
    # initiate the environment, get action and state space size, and get action range
    if "v2" not in environment:
        if environment =="point":
            env=gym.make("PointUMaze-v1")
            test_env=gym.make("PointUMaze-v1")
            num_actions=env.action_space.shape[0]
            num_states=env.observation_space.shape[0]
            action_range=np.array((1,0.25))
            density_estimator=SEstimator(1,20,20,[-2,-2])
            threshold=0.6
        elif environment=="maze":
            env=Env(n=max_steps,maze_type='square_large')
            test_env=Env(n=max_steps,maze_type='square_large')
            num_actions = env.action_size
            num_states  = env.state_size*2
            action_range=np.array((env.action_range,env.action_range))
            density_estimator=SEstimator(1,10,10,[-0.5,-0.5])
            threshold=0.15
        elif environment=="push":
            env=gym.make("PointPush-v1")
            test_env=gym.make("PointPush-v1")
            num_actions=env.action_space.shape[0]
            num_states=env.observation_space.shape[0]
            action_range=np.array((1,0.25))
            density_estimator=SEstimator(1,28,28,[-14,-2])
            threshold=0.6
        else:
            raise ValueError("The environment does not exist")
    else:
        env=FetchWrapper(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[environment+"-goal-observable"](render_mode="rgb_array",seed=seed))
        test_env=FetchWrapper(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[environment+"-goal-observable"](render_mode="rgb_array",seed=seed))
        num_actions=env.action_space.shape[0]
        num_states=env.observation_space.shape[0]
        threshold=None
        action_range=np.array((1,1,1,1))
        density_estimator=Model((-0.5,0.5),(0,1),(-0.5,0.5),0.02)

    # initiate the agent
    agent=Agent(num_actions,num_states,action_range,density_estimator,environment,threshold,model_avb,seed)
    
    # train the agent
    train(agent,env,address,environment,test_env)


if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-a','--address',required=True)
    # parser.add_argument('-e','--environment',required=True)
    # parser.add_argument('-m','--model',required=True)
    # args = parser.parse_args()

    # # set random seeds
    # seed=random.randint(0,1000)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # random.seed(seed)
    # np.random.seed(seed)
    
    
    # # save the seed to reproduce the results
    # with open(args.address+'/seed.txt', 'w') as f:
    #     f.write('%d' % seed)
    
    # if  not os.path.exists(args.address):
    #     sys.exit("The path is wrong!")
    # else:
    #     print("path is valid with seed: "+str(seed))
    
    # # train the agent
    # main(args.address,args.environment,args.model=="True",seed)

    with open("our_method/results/window-open/agent1/success_rates", 'rb') as fp:
        success=pickle.load(fp)
    
    plt.plot(success)
    plt.savefig("test")