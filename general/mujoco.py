import gym
import mujoco_maze  # noqa
import numpy as np
from mujoco_maze.point import PointEnv
from typing import Optional, Tuple


#env = gym.make("PointUMaze-v1")

planning_env=PointEnv("/tmp/tmp20ytwuz_.xml")

# s=env.reset()
# print("state: "+str(s))
# action=env.action_space.sample()
# print("action: "+str(action))
# s,r,done,_=env.step(action)
# print("next state: "+str(s))


state= np.array([ 0.06257095,  0.0572316,   0.06576819,  0.01252676, -0.05761685, -0.10378261])

action= [ 0.8605198,-0.1765576]
next_state=[ 0.91806567, -0.03906232, -0.11286506,  0.01253587, -0.05761787, -0.10378261]


def step(env,state, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
    qpos = state[0:3]
    qpos[2] += action[1]
    # Clip orientation
    if qpos[2] < -np.pi:
        qpos[2] += np.pi * 2
    elif np.pi < qpos[2]:
        qpos[2] -= np.pi * 2
    ori = qpos[2]
    # Compute increment in each direction
    qpos[0] += np.cos(ori) * action[0]
    qpos[1] += np.sin(ori) * action[0]
    qvel = np.clip(state[3:], -env.VELOCITY_LIMITS, env.VELOCITY_LIMITS)
    env.set_state(qpos, qvel)
    for _ in range(0, env.frame_skip):
        env.sim.step()
    next_obs = env._get_obs()
    return next_obs, 0.0, False, {}


next,r,done,_=step(planning_env,state,action)
print(next)


