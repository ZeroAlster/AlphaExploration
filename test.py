import gym
import mujoco_maze


# mountain-car is one of the things we can try


env = gym.make("Ant4Rooms-v0")
state=env.reset()