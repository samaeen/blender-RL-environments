import bpy
import numpy as np
import math
from math import radians
import time
import random


import gym
from gym import logger, spaces
from gym.envs.classic_control import utils



class CartPole3D:
    def __init__(self, sutton_barto_reward: bool = False, render_mode: str | None = None):
        self._sutton_barto_reward = sutton_barto_reward
        # --- Clean Scene safely ---
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        # --- Define sizes like Gym render code ---
        cartwidth, cartheight = 2.0, 1.0
        polewidth, polelen = 0.5, 3.0
        axleoffset = cartheight / 4.0

        #  1. Set World Background to White
        world = bpy.data.worlds["World"]
        world.use_nodes = True
        bg = world.node_tree.nodes["Background"]
        bg.inputs[0].default_value = (1, 1, 1, 1)

        #  2. Create Plane (Ground)
        bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
        plane = bpy.context.active_object
        plane.name = "Ground"

        
        #  Add brown material to plane
        brown_mat = bpy.data.materials.new(name="Brown")
        brown_mat.use_nodes = True
        bsdf = brown_mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.36, 0.25, 0.20, 1)  # Brown
        plane.data.materials.append(brown_mat)

        # --- Create Cart ---
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, .25))
        self.cart = bpy.context.active_object
        self.cart.name = "Cart"
        self.cart.scale = (cartwidth/2, cartwidth/2, cartheight/2)

        black_mat = bpy.data.materials.new(name="Black")
        black_mat.use_nodes = True
        bsdf_black = black_mat.node_tree.nodes["Principled BSDF"]
        bsdf_black.inputs["Base Color"].default_value = (0, 0, 0, 1)  # Black
        self.cart.data.materials.append(black_mat)

        # --- Create Pole ---
        self.pole_z = cartheight + axleoffset
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, self.pole_z))
        self.pole = bpy.context.active_object
        self.pole.name = "Pole"
        self.pole.scale = (polewidth/2, polewidth/2, polelen)


        red_mat = bpy.data.materials.new(name="Red")
        red_mat.use_nodes = True
        bsdf_red = red_mat.node_tree.nodes["Principled BSDF"]
        bsdf_red.inputs["Base Color"].default_value = (1, 0, 0, 1)  # Red
        self.pole.data.materials.append(red_mat)

        #  Parent Pole to Cart
        self.pole.parent = self.cart

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 24


        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        options: dict | None = None,
    ):
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = np.random.uniform(-0.05, 0.05, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        
        cart = bpy.data.objects.get("Cart")
        pole = bpy.data.objects.get("Pole")
        
        if cart is None or pole is None:
            raise ValueError("Cart or Pole not found. Make sure they are created in __init__().")

        if self.state is None:
            return None

        #  Read state
        x =  self.state

        #  Move cart & rotate pole based on state
        cart.location.x = 10* x[0]     # cart moves along X
        pole.rotation_euler.y = 20* radians(x[2])  # pole rotates based on theta

        #  Keyframe motion
        frame = bpy.context.scene.frame_current
        cart.keyframe_insert(data_path="location", frame=frame)
        pole.keyframe_insert(data_path="rotation_euler", frame=frame)

        #  Advance frame
        bpy.context.scene.frame_set(frame + 1)
        print(f" Frame {frame}: Cart moved to {cart.location.x:.2f}, Pole rotated to {pole.rotation_euler.y:.2f} rad")


    def close(self):
        pass


#  MAIN USAGE EXAMPLE
if __name__ == "__main__":
    env = CartPole3D()
    states = env.observation_space.shape[0]
    actions = env.action_space.n

    print(states)

    episodes = 10
    current_episode = 1
    done = False
    score = 0

    def train():
        global current_episode, done, score, state

        if current_episode > episodes:
            print("Training finished!")
            return None  # stop timer

        # If starting a new episode
        if not done and score == 0:
            state, info = env.reset()
            done = False
            score = 0

        #  Run ONE STEP
        env.render()
        action = random.choice([0, 1])
        n_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        print(f"Episode {current_episode} | Step score: {score}")

        #  When episode ends, move to next one
        if done:
            print(f"🎯 Episode {current_episode} finished with score {score}")
            current_episode += 1
            done = False
            score = 0  # reset for next episode

        return 0.1  # call again after 0.1s

    #  Start training smoothly
    bpy.app.timers.register(train)
