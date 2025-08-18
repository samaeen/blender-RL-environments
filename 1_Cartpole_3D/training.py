import sys
sys.path.append("C:/Users/Shoummo/Documents/Project/BlenderRL/1_Cartpole_3d")

import gymnasium as gym
from cartpole import CartPole3D

env = CartPole3D()
print(env.reset())   # should work now



import cartpole
import gymnasium as gym   # <-- use gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

log_path = "C:/Users/Shoummo/Documents/Project/BlenderRL/1_Cartpole_3d/Logs"

env = DummyVecEnv([lambda: cartpole.CartPole3D()])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

print(env.observation_space.sample())

model.learn(total_timesteps=20000)


PPO_path="C:/Users/Shoummo/Documents/Project/BlenderRL/1_Cartpole_3d/SavedModels/CartpoleModel_1"

model.save(PPO_path)