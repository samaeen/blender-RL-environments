from cartpole import CartPole3D
import random
import bpy
from stable_baselines3 import PPO

if __name__ == "__main__":

    env = CartPole3D()
    PPO_path="C:/Users/Shoummo/Documents/Project/BlenderRL/1_Cartpole_3d/SavedModels/CartpoleModel_1"
    model = PPO.load(PPO_path, env=env)

    states = env.observation_space.shape[0]
    actions = env.action_space.n
    print(states)

    episodes = 10
    current_episode = 1
    done = False
    score = 0
    obs = None   # <-- initialize globally

    def train():
        global current_episode, done, score, obs   # <-- include obs

        if current_episode > episodes:
            print("Training finished!")
            return None  # stop timer

        # If starting a new episode
        if obs is None or (done and score == 0):
            obs, info = env.reset()
            done = False
            score = 0

        # Run ONE STEP
        env.render()
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        print(f"Episode {current_episode} | Step score: {score}")

        # When episode ends, move to next one
        if done:
            print(f" Episode {current_episode} finished with score {score}")
            current_episode += 1
            obs = None  # force reset on next call
            score = 0

        return 0.1  # call again after 0.1s

    # Start training smoothly
    bpy.app.timers.register(train)
