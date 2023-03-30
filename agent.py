import torch
from env import Env
import torch.nn as nn
import torch.nn.functional as F 
from collections import deque
import numpy as np
import random
import torch.optim as optim
from simple_estimator import SEstimator
from torch.autograd import Variable
import sys
import math

#hyper params
######################################
replay_buffer_size = 5e5
hidden_size=128
actor_learning_rate=1e-4
critic_learning_rate=6e-4
epsilon_decay=0.9999997
epsilon=1
noise_scale=0.3
RRT_budget=35
max_steps   = 50
minimum_exploration=0.01
rrt_prob=0.9
short_memory_size=int(5e4)
tau=1e-2
gamma=0.98
rrt_min_visit=1000
######################################


class Node:
    def __init__(self, parent,coordination):
        self.parent=parent
        self.coordination=coordination
        self.children=[]
        self.edges=[]
    
    def add_child(self,child):
        self.children.append(child)




class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.hit=0
        self.shuffle_interval=max_size/2
    
    def push(self, state, action, reward, next_state, done,step):
        experience = (state, action, np.array([reward]), next_state, np.array([done]),np.array([step]))
        self.buffer.append(experience)
        self.hit+=1
        if self.hit % self.shuffle_interval==0:
            random.shuffle(self.buffer)    


        # shuffle the buffer

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

    def __len__(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x
    



class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,action_range):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        self.action_range=action_range
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = torch.tanh(self.linear4(x))*self.action_range

        return x


class Agent():
    def __init__(self,num_actions,num_states,action_range,hidden_size=hidden_size, actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, gamma=gamma, tau=tau,memory_size=int(replay_buffer_size),epsilon=epsilon,epsilon_decay=epsilon_decay):
        
        # Params
        self.num_actions = num_actions
        self.num_states  = num_states
        self.gamma = gamma
        self.tau = tau
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.action_range=action_range
        self.short_memory_updates=0
        self.noise_scale=noise_scale

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions,action_range)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions,action_range)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, 1)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training, density estimator, memory, and path_array
        self.density_estimator=SEstimator(1,10,10,[-0.5,-0.5])
        self.memory = Memory(memory_size)
        self.short_memory=Memory(short_memory_size)
        self.path_array=[[[] for i in range(self.density_estimator.visits.shape[0])] for j in range(self.density_estimator.visits.shape[1])]        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    
    def RRT(self,coordination):
        success=False
        goal=None
        nodes=[]
        root=Node(None,coordination)
        nodes.append(root)
        env=Env(n=max_steps,maze_type='square_large')
        
        # create the grapgh
        for _ in range(RRT_budget):
            
            # sampling the node from the tree and choosing a random action
            node=random.choice(nodes)
            action=np.random.uniform(-self.action_range,self.action_range,size=(2,))
            env._state['state']=env.to_tensor(node.coordination[0:2])
            new_coordination,_,_,_=env.step(action)
            
            # creating the new node and adding it to the tree
            child=Node((node,action),new_coordination)
            node.add_child((child,action))
            nodes.append(child)
            
            # check if we hit a non-visited region
            cell_x,cell_y=self.point_to_cell(new_coordination)
            if self.density_estimator.visits[cell_x][cell_y] <rrt_min_visit or env.is_success:
                success=True
                goal=child
                break
        
        
        # if we do not have the goal, we randomly select one of the nodes as the goal.
        if not success:
            goal=random.choice(nodes)
            while goal==root:
                goal=random.choice(nodes)
        
        # find a path from root to goal or randomly selected node
        option=[]
        node=goal
        while node.parent is not None:
            option.append(node.parent[1])
            node=node.parent[0]
        return option    
    
    
    def get_action(self, state,warmup,evaluation=False):
        
        if random.uniform(0, 1)<self.epsilon and not warmup and not evaluation:
            # we will output an option by RRT or a random action
            if random.uniform(0, 1)<rrt_prob:
                option= self.RRT(state)
            else:
                option=[np.random.uniform(-self.action_range,self.action_range,size=(2,))]
        else:
            #get the primitive action from the network
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))
            action = self.actor.forward(state)
            action = action.detach().numpy().flatten()
            
            # disable during evaluation
            if not evaluation:
                noise=np.random.normal(0, self.noise_scale, size=self.num_actions).clip(-self.action_range, self.action_range)
                action=np.clip(action+noise,-self.action_range, self.action_range)
            
            option=[action]
        
        # reverse the option
        option.reverse()

        # update the epsilon
        if self.epsilon>minimum_exploration:
            self.epsilon=self.epsilon*self.epsilon_decay

        return option
    
    
    def update(self, batch_size,update_number):
        
        if update_number<self.short_memory_updates and len(self.short_memory)>0:
            states, actions, rewards, next_states, done,steps = self.short_memory.sample(min(batch_size,len(self.short_memory)))
        else:
            states, actions, rewards, next_states, done,steps = self.memory.sample(batch_size)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        done=torch.FloatTensor(np.array([1-i for i in done]))
        steps=torch.FloatTensor(np.array(steps))
        
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions)
        Qprime = rewards + (done* torch.pow(self.gamma,steps) *next_Q).detach()
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