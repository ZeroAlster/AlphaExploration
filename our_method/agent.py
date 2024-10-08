import torch
from general.maze import Env
import torch.nn as nn
import torch.nn.functional as F 
from collections import deque
import numpy as np
import random
import torch.optim as optim
from general.simple_estimator import SEstimator
from general.graph import Graph
from torch.autograd import Variable
import math
import sys
import gym 
import mujoco_maze  # noqa
import mujoco_maze.maze_env
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
from wrappers import FetchWrapper
import copy

#hyper params
######################################
replay_buffer_size = 1e6
hidden_size=128
actor_learning_rate=4e-4
critic_learning_rate=1e-3
epsilon_decay=0.999998
epsilon=1
RRT_budget=60
max_steps= 100
short_memory_size=int(5e4)
tau=1e-2
gamma=0.99
minimum_exploration=0.01
######################################



obs_range={
    "button-press-topdown-v2":[(-0.4,0.4),(0.2,1),(-0.2,0.5)],
    "soccer-v2":[(-0.3,0.3),(0.3,1),(0,0.5)],
    "window-open-v2":[(-0.1,0.5),(0.3,1),(-0.1,0.5)]
}

noise_scale=[0.15,0.15,0.15,0.15]


# cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Node:
    def __init__(self, parent,coordination,data=None):
        self.parent=parent
        self.coordination=coordination
        self.data=data

class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.hit=0
        self.shuffle_interval=max_size/2
    
    def push(self, state, action, reward, next_state, done,step):
        experience = (state, action, np.array([reward]), next_state, np.array([done]),np.array([step]))
        self.buffer.append(experience)
        self.hit+=1
        
        # shuffle the buffer
        # if self.hit % self.shuffle_interval==0:
        #     random.shuffle(self.buffer)    


    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        step_batch=[]

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done,step = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            step_batch.append(step)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch,step_batch
    

    def CER_sample(self,batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        step_batch=[]

        batch=[]
        for i in range(batch_size):
            batch.append(self.buffer[-(i+1)])

        for experience in batch:
            state, action, reward, next_state, done,step = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            step_batch.append(step)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch,step_batch


    def __len__(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,init_w=3e-3):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)


        # initialize the last layer like DDPG main paper
        # self.linear4.weight.data.uniform_(-init_w, init_w)
        # self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """

        if len(state.shape)==3:
            x = torch.cat([state, action], 2)
        elif len(state.shape)==2:
            x = torch.cat([state, action], 1)
            
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
    

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,action_range,init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        self.action_range=action_range


        # initialize the last layer like DDPG main paper
        # self.linear4.weight.data.uniform_(-init_w, init_w)
        # self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        if self.action_range[0]!=self.action_range[1]:
            x = torch.tanh(self.linear4(x))*torch.tensor((self.action_range[0],self.action_range[1]),device="cuda")
        else:
            x=torch.tanh(self.linear4(x))*torch.tensor(self.action_range,device='cuda')
        return x.float()


class Agent():
    def __init__(self,num_actions,num_states,action_range,density_estimator,environment,threshold,model_avb,seed,
                 hidden_size=hidden_size, actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, 
                 gamma=gamma, tau=tau,memory_size=int(replay_buffer_size),epsilon=epsilon,epsilon_decay=epsilon_decay):
        
        # Params
        self.num_actions = num_actions
        self.num_states  = num_states
        self.gamma = gamma
        self.tau = tau
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.action_range=action_range
        self.short_memory_updates=0
        self.simulator=environment
        self.threshold=threshold
        self.model_access=model_avb


        # define simulator environment and the graph if model is not available
        if self.simulator=="maze":
            self.model=Env(n=max_steps,maze_type='square_large')
            self.graph=Graph(0.1,[-0.5,-0.5],10)
        elif self.simulator=="point":
            self.model=gym.make("PointUMaze-v1")
            self.graph=Graph(0.2,[-2,-2],20)
        elif self.simulator=="push":
            self.model=gym.make("PointPush-v1")
            self.graph=Graph(0.2,[-14,-2],28)
        elif "v2" in self.simulator:
            self.graph=density_estimator
            self.model=FetchWrapper(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[environment+"-goal-observable"](render_mode="rgb_array",seed=seed))
        else:
            sys.exit("simulator is not valid.")

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions,action_range)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions,action_range)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, 1)

        # copy networks on GPU
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training, density estimator, memory, and path_array
        self.density_estimator=density_estimator
        self.memory = Memory(memory_size)
        self.short_memory=Memory(short_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    
    def neighbour(self,state,goal):
        if np.linalg.norm(state-goal)<=self.threshold:
            return True
        else:
            return False
        
    def step(self,node,action):

        if "v2" not in self.simulator:
            if self.simulator=="maze":
                self.model._state['state']=self.model.to_tensor(node.coordination[0:2])
                new_coordination,_,_,_=self.model.step(action)
            elif self.simulator=="point" or self.simulator=="push":
                new_coordination,_,_,_=self.model.planning_step(node.coordination,action)
            else:
                sys.exit("wrong simulator!")
        else:
            new_coordination,_,_,_=self.model.step(action,sim_state=node.data)
        
        return new_coordination


    def obs_clipping(self,coordination):

        bounds=obs_range[self.simulator]

        if (coordination[0]<bounds[0][0] or coordination[0]>bounds[0][1]):
            return True
        if (coordination[1]<bounds[1][0] or coordination[1]>bounds[1][1]):
            return True
        if (coordination[2]<bounds[2][0] or coordination[2]>bounds[2][1]):
            return True        
        
        return False

    def RRT(self,coordination):
        nodes=[]

        if "v2" in self.simulator:
            root=Node(None,coordination,data=copy.deepcopy(self.model.data))
        else:
            root=Node(None,coordination)
        
        nodes.append(root)
        goal=root
        
        # create the graph
        for _ in range(RRT_budget):
            
            # sampling the node from the tree and choosing a random action
            node=random.choice(nodes)
            
            action=np.random.uniform(-self.action_range,self.action_range,size=(len(self.action_range),))
            model_response=self.step(node,action)
            # skip out of range states
            # if model_response[1]>self.model.observation_space.high[1] or model_response[1]<self.model.observation_space.low[1] or model_response[0]<self.model.observation_space.low[0] or model_response[0]>self.model.observation_space.high[0]:
            #   continue

            if self.model_access:
                new_coordination=model_response
            else:
                dic_obs,dic_action=self.graph.find(node.coordination)
                if dic_obs is not None:
                    new_coordination=dic_obs
                    action=dic_action    
                else:
                    if "v2" not in self.simulator:
                        new_coordination=model_response
                        child=Node((node,action),new_coordination)
                        nodes.append(child)
                        goal=child
                    else:
                        goal=root
                    break
            
            # observation clipping
            if self.obs_clipping(new_coordination):
                continue
            
            # creating the new node and adding it to the tree
            child=Node((node,action),new_coordination,data=copy.deepcopy(self.model.data))
            nodes.append(child)

            # check if we hit the goal
            # if self.neighbour(new_coordination[0:2],new_coordination[-2:]):
            if self.model.success:
                goal=child
                break

            if "v2" not in self.simulator:
                # update the goal to be the node with minimum visit
                cell_x,cell_y=self.point_to_cell(new_coordination)
                goal_cell_x,goal_cell_y=self.point_to_cell(goal.coordination)
                if self.density_estimator.visits[cell_x][cell_y] <self.density_estimator.visits[goal_cell_x][goal_cell_y]:
                    goal=child
            else:
                if self.graph.get_density(new_coordination)<=self.graph.get_density(goal.coordination):
                    goal=child   
            
        
        # if the goal is the root, randomly select one of the other nodes 
        if goal==root and len(nodes)>1:
            goal=random.choice(nodes)
            while goal==root:
                goal=random.choice(nodes)        
        
        # find a path from root to the goal or the randomly selected node
        option=[]
        node=goal
        while node.parent is not None:
            option.append(node.parent[1])
            node=node.parent[0]

        if goal==root and len(nodes)==1:
            option.append(np.random.uniform(-self.action_range,self.action_range,size=(len(self.action_range),)))

        return option        
    
    # main function
    # def get_action(self, state,warmup,evaluation=False,data=None):
        
    #     if random.uniform(0, 1)<self.epsilon and not warmup and not evaluation:
            
    #         exploration=True
            
    #         # we will output an option by RRT or a random action
    #         self.model.data=data
    #         option= self.RRT(state)

    #         # record option length for distribution in the tree
    #         length=len(option)

    #         # to see the impact of rrt exploration 
    #         #option=[np.random.uniform(-self.action_range,self.action_range,size=(2,))]
    #     else:            
            
    #         exploration =False
    #         length=1

    #         #get a primitive action from the network
    #         state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(device)
    #         action = self.actor.forward(state)
    #         action = action.cpu().detach().numpy().flatten()
    #         option=[action]
        
    #     # reverse the option
    #     option.reverse()

    #     # update the epsilon
    #     if self.epsilon> minimum_exploration:
    #         self.epsilon=self.epsilon*self.epsilon_decay

    #     return option,exploration,length

    # noisy action
    def get_action(self, state,warmup,evaluation=False,data=None):
        
        #get a primitive action from the network
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(device)
        action = self.actor.forward(state)
        action = action.cpu().detach().numpy().flatten()
        
        # adding noise to the action if it is not evaluation
        if not evaluation:
            noise=np.ones(4)
            noise[0]=np.random.normal(0, noise_scale[0], size=1).clip(-self.action_range[0], self.action_range[0])
            noise[1]=np.random.normal(0, noise_scale[1], size=1).clip(-self.action_range[1], self.action_range[1])
            noise[2]=np.random.normal(0, noise_scale[2], size=1).clip(-self.action_range[1], self.action_range[1])
            noise[3]=np.random.normal(0, noise_scale[3], size=1).clip(-self.action_range[1], self.action_range[1])
            action=np.clip(action+noise,-self.action_range, self.action_range)

        option=[action]

        return option,False,1
    
    
    def update(self, batch_size,update_number):
        
        if update_number<self.short_memory_updates and len(self.short_memory)>0:
            states, actions, rewards, next_states, done,steps = self.short_memory.sample(min(batch_size,len(self.short_memory)))
        else:
            states, actions, rewards, next_states, done,steps = self.memory.sample(batch_size)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        done=torch.FloatTensor(np.array([1-i for i in done])).to(device)
        steps=torch.FloatTensor(np.array(steps)).to(device)

        
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions)
        if len(next_Q.shape)==3:
            next_Q=torch.reshape(next_Q, (128,1,8))
        Qprime = rewards + (done* torch.pow(self.gamma,steps) *next_Q).detach()
        if len(Qprime.shape)==3:
            Qprime=Qprime.sum(axis=2)/torch.count_nonzero(Qprime, dim=2) 
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    

    def point_to_cell(self,coordination):
        cell_x=math.floor((coordination[0]-self.density_estimator.env_start[0])/self.density_estimator.cell_side)
        cell_y=math.floor((coordination[1]-self.density_estimator.env_start[1])/self.density_estimator.cell_side)

        cell_x=min(cell_x,self.density_estimator.visits.shape[0]-1)
        cell_x=max(cell_x,0)
        cell_y=min(cell_y,self.density_estimator.visits.shape[1]-1)
        cell_y=max(cell_y,0)

        return cell_x,cell_y