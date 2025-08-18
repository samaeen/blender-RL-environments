import sys
sys.path.append("C:/Users/Shoummo/Documents/Project/BlenderRL/2_pelletGrabber")

import gymnasium as gym
from pelletGrabber_V2 import pelletGrabberEnv

env = pelletGrabberEnv()
print(env.reset())   # should work now


import gymnasium as gym   # <-- use gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

log_path = "C:/Users/Shoummo/Documents/Project/BlenderRL/2_pelletGrabber/Logs"

env = DummyVecEnv([lambda: pelletGrabberEnv()])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)


model.learn(total_timesteps=100000)


PPO_path="C:/Users/Shoummo/Documents/Project/BlenderRL/2_pelletGrabber/SavedModels/pelletGrabberModel_1"

model.save(PPO_path)