import gym
import mujoco_maze  # noqa
env = gym.make("PointUMaze-v1")



s=env.reset()

print(s)



