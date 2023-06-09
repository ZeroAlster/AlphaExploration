import torch
from general.maze import Env
import torch.nn as nn
import torch.nn.functional as F 
from collections import deque
import numpy as np
import random
import torch.optim as optim
from general.simple_estimator import SEstimator
from torch.autograd import Variable
import math
import sys

#hyper params
######################################
replay_buffer_size = 5e5
hidden_size=128
actor_learning_rate=1e-4
critic_learning_rate=1e-3
epsilon_decay=0.9999988
epsilon=1
RRT_budget=40
max_steps   = 100
short_memory_size=int(5e4)
tau=1e-2
gamma=0.99
minimum_exploration=0.01
######################################


# cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Node:
    def __init__(self, parent,coordination):
        self.parent=parent
        self.coordination=coordination

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
        if self.hit % self.shuffle_interval==0:
            random.shuffle(self.buffer)    


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
        self.density_estimator=SEstimator(1,10,10,[-0.5,-0.5])
        self.memory = Memory(memory_size)
        self.short_memory=Memory(short_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    
    def RRT(self,coordination):
        nodes=[]
        root=Node(None,coordination)
        nodes.append(root)
        env=Env(n=max_steps,maze_type='square_large')
        goal=root

        
        # create the grapgh
        for _ in range(RRT_budget):
            
            # sampling the node from the tree and choosing a random action
            node=random.choice(nodes)
            action=np.random.uniform(-self.action_range,self.action_range,size=(2,))
            env._state['state']=env.to_tensor(node.coordination[0:2])
            new_coordination,_,_,_=env.step(action)
            
            # creating the new node and adding it to the tree
            child=Node((node,action),new_coordination)
            nodes.append(child)

            # check if we hit the goal
            if env.is_success:
                goal=child
                break

            # update the goal to be the node with minimum visit
            cell_x,cell_y=self.point_to_cell(new_coordination)
            goal_cell_x,goal_cell_y=self.point_to_cell(goal.coordination)
            if self.density_estimator.visits[cell_x][cell_y] <=self.density_estimator.visits[goal_cell_x][goal_cell_y]:
                goal=child
        
        
        # if the goal is the root, randomly select one of the other nodes 
        if goal==root:
            goal=random.choice(nodes)
            while goal==root:
                goal=random.choice(nodes)
        
        # find a path from root to the goal or the randomly selected node
        option=[]
        node=goal
        while node.parent is not None:
            option.append(node.parent[1])
            node=node.parent[0]
        return option    
    
    
    def get_action(self, state,warmup,evaluation=False):
        
        if random.uniform(0, 1)<self.epsilon and not warmup and not evaluation:
            
            # we will output an option by RRT or a random action
            option= self.RRT(state)

            # to see the impact of rrt exploration 
            # option=[np.random.uniform(-self.action_range,self.action_range,size=(2,))]
        else:
            #get a primitive action from the network
            state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(device)
            action = self.actor.forward(state)
            action = action.cpu().detach().numpy().flatten()
            option=[action]
        
        # reverse the option
        option.reverse()

        # update the epsilon
        if self.epsilon> minimum_exploration:
            self.epsilon=self.epsilon*self.epsilon_decay

        return option
    
    
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