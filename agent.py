import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import sys



random.seed(10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyper params
###################################### 
epsilon = 1
epsilon_decay = 0.995
min_epsilon = 0.001
###################################### 


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self,episode_rollout,reward):
        
        for sample in episode_rollout:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)

        for sample in episode_rollout:
            action=sample[1]
            if isinstance(sample[1][0], torch.Tensor):
                action=(sample[1][0].cpu(),sample[1][1].cpu())
            self.buffer[self.position] = (sample[0].cpu(), action, reward.cpu(), sample[2].cpu(), sample[3])
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done =  map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size,action_range, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range=action_range
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample()
        action = torch.tanh(mean+ std*z.to(device))*self.action_range
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(device)
        action = torch.tanh(mean + std*z)*self.action_range
        
        action  = action.to(device)
        return action[0]


class Agent():
    def __init__(self,density_model,replay_buffer, policy_network,soft_network1,soft_network2,value_network,target_network):
        self.density_model=density_model
        self.replay_buffer=replay_buffer
        self.policy_network=policy_network
        self.soft_network1=soft_network1
        self.soft_network2=soft_network2
        self.value_network=value_network
        self.target_network=target_network
        self.value_criterion=nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.value_lr  = 3e-4
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.value_optimizer  = optim.Adam(self.value_network.parameters(), lr=self.value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_network1.parameters(), lr=self.soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_network2.parameters(), lr=self.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.policy_lr)

    def update(self,batch_size,gamma=0.99,soft_tau=1e-2,):
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_network1(state, action)
        predicted_q_value2 = self.soft_network2(state, action)
        predicted_value    = self.value_network(state)
        new_action, log_prob, epsilon, mean, log_std = self.policy_network.evaluate(state)
    
    # Training Q Function
        target_value = self.target_network(next_state)
        target_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()   
        
    # Training Value Function
        predicted_new_q_value = torch.min(self.soft_network1(state, new_action),self.soft_network2(state, new_action))
        target_value_func = predicted_new_q_value - log_prob
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    # Training Policy Function
        policy_loss = (log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        for target_param, param in zip(self.target_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


    def exploration(self,cur_state,env):
        global epsilon

        if np.random.uniform(0,1) > epsilon:               
        
            action = self.policy_network.get_action(cur_state).detach()
            
            env.step(action)
            next_state = env.state
            self.density_model.increment(env.state)
            reward = env.reward(self.density_model.prob(env.state))
            done=env.is_done

        else:
            dx=random.uniform(-0.95,0.95)
            dy=random.uniform(-0.95,0.95)

            action = (dx,dy)
            
            env.step(action)
            
            next_state = env.state
            self.density_model.increment(env.state)
            reward = env.reward(self.density_model.prob(env.state))
            done=env.is_done
        
        if epsilon > 0.001:
            epsilon*=epsilon_decay

        return action,next_state,reward,done