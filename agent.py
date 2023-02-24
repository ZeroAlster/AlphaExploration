import pickle
from stable_baselines3.common.callbacks import BaseCallback
from env import Env


#hyper params
######################################
max_steps   = 50
evaluation_attempts=5
######################################


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self,address, verbose=0,checkpoint=10000,episode_length=50):
        super(CustomCallback, self).__init__(verbose)
        self.checkpoint=checkpoint
        self.locations=[]
        self.success_rates=[]
        self.path=address
        self.len_episode=episode_length

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        
        if self.num_timesteps % self.len_episode== self.len_episode-1:
            self.locations.append(self.locals["new_obs"])
        
        if self.num_timesteps % self.checkpoint==0:
            print("next checkpoint: "+str(self.num_timesteps)+"  steps")
            env=Env(n=max_steps,maze_type='square_large')
            success=0
            for _ in range(evaluation_attempts):
                obs = env.reset()
                done=False
                while not done:
                    action, _states=self.model.predict(obs, deterministic=False)
                    obs,_, done,_= env.step(action)
                if env.is_success:
                    success+=1
            self.success_rates.append(success/evaluation_attempts)



        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        with open(self.path+"/locations", "wb") as fp:
            pickle.dump(self.locations, fp)

        with open(self.path+"/success_rates", "wb") as fp:
            pickle.dump(self.success_rates, fp)