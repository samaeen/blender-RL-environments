from pelletGrabber_V2 import pelletGrabberEnv
import random
import bpy
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = pelletGrabberEnv()
    PPO_path = "C:/Users/Shoummo/Documents/Project/BlenderRL/2_pelletGrabber/SavedModels/pelletGrabberModel_1"
    model = PPO.load(PPO_path, env=env)

    states = env.observation_space.shape[0]
    actions = env.action_space.n
    print(f"Observation space: {states} | Action space: {actions}")

    episodes = 10
    current_episode = 1
    done = True
    episode_score = 0
    total_score = 0
    obs = None

    def train():
        global current_episode, done, episode_score, total_score, obs

        if current_episode > episodes:
            print(f"\n Training finished! Total score across all episodes: {total_score}")
            return None

        if done:
            obs, info = env.reset()
            done = False
            episode_score = 0
            print(f"\n--- Starting Episode {current_episode} ---")

        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_score += reward
        total_score += reward
        done = terminated or truncated

        print(f"Episode {current_episode} | Action: {action} | Reward: {reward} | "
              f"Episode Total: {episode_score} | Overall Total: {total_score}")

        if done:
            print(f" Episode {current_episode} finished with score {episode_score}")
            current_episode += 1

        return 0.1

    # Start training smoothly
    bpy.app.timers.register(train)
