import torch
from env import Env
from simple_estimator import SEstimator
from agent import ValueNetwork
from agent import SoftQNetwork
from agent import PolicyNetwork
from agent import ReplayBuffer
from agent import Agent
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import sys

#hyper params
######################################
hidden_dim = 256
replay_buffer_size = 1000000
max_frames  = 400000
max_steps   = 50
batch_size  = 128
action_range=0.95
######################################


# cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot(frame_idx, rewards,destinations,env):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.savefig("figures/rewards.png")
    
    _, ax = plt.subplots(1, 1, figsize=(5, 4))
    for x, y in env.maze._walls:
        ax.plot(x, y, 'k-')
    for sample in destinations:
        ax.plot(sample[0],sample[1],marker='o',markersize=2,color="red")
    plt.savefig("figures/destinations.png")



def train(agent,env):
    frame_idx=1
    rewards= []
    destinations=[]
    done=False
    while frame_idx < max_frames and not done:
        env.reset()
        state=env.state
        terminal_reward=0
        terminal_state=None
        episode_rollout=[]
        
        for _ in range(max_steps):

            action,next_state,reward,done = agent.exploration(torch.cat([state,env.goal],dim=0),env)
            episode_rollout.append([torch.cat([state,env.goal],dim=0),action,torch.cat([next_state,env.goal],dim=0),done])
            
            state = next_state
            frame_idx += 1
            
            terminal_state=state
            terminal_reward = reward
            
            if env.is_success:
                print("we found the goal!")
                break

        # add episode samples to the replay buffer
        agent.replay_buffer.push(episode_rollout,terminal_reward)
        #print("eipsode is complete!")

        # update the network
        if frame_idx % 201==0:
            agent.update(batch_size)
            print("update is done!")

        # get the rewards and terminal states of epidoes
        rewards.append(terminal_reward)
        destinations.append(terminal_state)

        # plot the rewards and episodes terminal states
        if frame_idx % 1001 == 0:
            print("plotting...")
            plot(frame_idx, rewards,destinations,env)





def main():
    # initialize the environment and get the dimensions: state is the concatenation of the current and goal states
    env=Env(n=100,maze_type='square_large')
    action_dim = env.action_size
    state_dim  = env.state_size*2
    
    # initiate the metworks of agent
    value_net= ValueNetwork(state_dim, hidden_dim).to(device)
    target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)
    soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim,action_range).to(device)
    
    # copy parameters of value network to target network
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)
        pass
    
    # define density model and replay_buffer
    density_model=SEstimator(1,10,10,[-0.5,-0.5])
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # instantiate the agent
    agent=Agent(density_model,replay_buffer,policy_net,soft_q_net1,soft_q_net2,value_net,target_value_net)

    # train the agent
    train(agent,env)


if __name__ == '__main__':
    main()