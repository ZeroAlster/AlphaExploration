import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import deque
import numpy as np
import random
import torch.optim as optim
from torch.autograd import Variable

#hyper params
######################################
replay_buffer_size = 1e6
hidden_size=128
actor_learning_rate=1e-4
critic_learning_rate=1e-3
max_steps= 100
tau=1e-2
gamma=0.99
minimum_exploration=0.01
noise_scale=0.35
######################################


# cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    def __init__(self,num_actions,num_states,action_range,hidden_size=hidden_size, actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, gamma=gamma, tau=tau,memory_size=int(replay_buffer_size)):
        
        # Params
        self.num_actions = num_actions
        self.num_states  = num_states
        self.gamma = gamma
        self.tau = tau
        self.action_range=action_range
        self.noise_scale=noise_scale

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

        # Training, and memory
        self.memory = Memory(memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        
    
    def get_action(self, state,evaluation=False):
        
        #get a primitive action from the network
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(device)
        action = self.actor.forward(state)
        action = action.cpu().detach().numpy().flatten()
        
        # adding noise to the action if it is not evaluation
        if not evaluation:
            noise=np.random.normal(0, self.noise_scale, size=self.num_actions).clip(-self.action_range, self.action_range)
            action=np.clip(action+noise,-self.action_range, self.action_range)

        option=[action]

        return option
    
    
    def update(self, batch_size):
        
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