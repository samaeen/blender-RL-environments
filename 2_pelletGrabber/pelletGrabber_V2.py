from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import bpy
import math
from mathutils import Vector, Euler

class pelletGrabberEnv(Env):
    def __init__(self):
        # --- RL Parameters ---
        self.action_space = Discrete(4)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # --- Robot Parameters ---
        self.wheel_speed = 2   # MUCH smaller step for smooth motion

        # Get robot
        self.robot = bpy.data.objects.get("Robot")
        if self.robot is None:
            raise Exception("No object named 'Robot' found!")

        # Materials for rewards
        self.greenMat = bpy.data.materials.new(name="Green")
        self.greenMat.use_nodes = True
        self.greenMat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0, 1, 0, 1)

        self.redMat = bpy.data.materials.new(name="Red")
        self.redMat.use_nodes = True
        self.redMat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 0, 0, 1)



        # State
        self.state = [0, 0, 0, 10, 10, 10, 10, 10]
        self.current_action = 2  # start by moving forward

        # Start the continuous simulation
        #bpy.app.timers.register(self.robotMove)
    # ===================================================
    # Gym API
    # ===================================================
    def step(self, action):
        # Move robot + handle collisions through render()
        reward = self.render(action)

        print(reward)
        terminated = reward != 0 
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self):
        """Reset robot & target"""
        self.robot.location = Vector((0, 0, self.robot.dimensions.z/2))
        self.robot.rotation_euler.z = 0.0

        if hasattr(self, "target") and self.target:
            bpy.data.objects.remove(self.target, do_unlink=True)
            self.target = None


        self.generateTarget()

        self.state = [self.robot.location.x, self.robot.location.y, self.robot.rotation_euler.z,
                      10.0, 10.0, 10.0, 10.0, 10.0]
        return np.array(self.state, dtype=np.float32), {}

    def render(self, action=None, mode="human"):
        if action is not None:
            wheel_base = self.robot.dimensions.y
            if action == 0:       # turn left
                v_l, v_r = -self.wheel_speed, self.wheel_speed
            elif action == 1:     # turn right
                v_l, v_r = self.wheel_speed, -self.wheel_speed
            elif action == 2:     # forward
                v_l, v_r = self.wheel_speed, self.wheel_speed
            elif action == 3:     # backward
                v_l, v_r = -self.wheel_speed, -self.wheel_speed

            v = (v_r + v_l) / 2.0
            omega = (v_r - v_l) / wheel_base

            yaw = self.robot.rotation_euler.z
            self.robot.location.x += v * math.cos(yaw)
            self.robot.location.y += v * math.sin(yaw)
            self.robot.rotation_euler.z += omega

        plane = bpy.data.objects.get("Plane")
        target_collection = bpy.data.collections.get("Targets")
        wall_collection = bpy.data.collections.get("Walls")
        if not target_collection or not wall_collection:
            raise Exception("Collection 'Targets' or 'Walls' not found!")

        num_rays = 5
        arc_angle = math.radians(30)
        start_angle = -arc_angle / 2
        max_distance = 2

        hit_reward = 0
        ray_slots = [max_distance] * num_rays
        target_hit = False

        forward_dir = self.robot.matrix_world.to_quaternion() @ Vector((0, 1, 0))
        start = self.robot.matrix_world.translation + (self.robot.matrix_world.to_quaternion() @ Vector((0, 1.05, 0))) * 1.0

        for i in range(num_rays):
            angle = start_angle + (i / (num_rays - 1)) * arc_angle
            rotation = Euler((0, 0, angle), 'XYZ')
            ray_dir = rotation.to_matrix() @ forward_dir

            depsgraph = bpy.context.evaluated_depsgraph_get()
            result, location, normal, index, obj, matrix = bpy.context.scene.ray_cast(
                depsgraph,
                start,
                ray_dir,
                distance=max_distance
            )

            if result:
                if obj.name in [hit_obj.name for hit_obj in target_collection.objects]:
                    # ✅ HIT TARGET FIRST → STOP IMMEDIATELY
                    if not plane.data.materials:
                        plane.data.materials.append(self.greenMat)
                    else:
                        plane.data.materials[0] = self.greenMat

                    bpy.data.objects.remove(obj, do_unlink=True)
                    self.target = None
                    hit_reward = +5
                    target_hit = True
                    print(target_hit)
                    break   # 🚨 STOP scanning — don’t overwrite reward

                elif obj.name in [hit_obj.name for hit_obj in wall_collection.objects]:
                    # ✅ HIT WALL FIRST → STOP IMMEDIATELY
                    if not plane.data.materials:
                        plane.data.materials.append(self.redMat)
                    else:
                        plane.data.materials[0] = self.redMat
                    print("wall hit")
                    hit_reward = -1
                    break   # 🚨 STOP scanning — don’t overwrite reward

            else:
                ray_slots[i] = max_distance

        # ✅ Spawn new target after the loop (only if we hit one)
        if target_hit:
            self.generateTarget()

        # Update state
        robot_x, robot_y, robot_yaw = (
            self.robot.location.x,
            self.robot.location.y,
            self.robot.rotation_euler.z
        )
        self.state = [robot_x, robot_y, robot_yaw, *ray_slots]

        return hit_reward


    def generateTarget(self):
        locationX = random.randint(0, 10)
        locationY = random.randint(0, 10)

        target_collection = bpy.data.collections.get("Targets")
        if target_collection is None:
            target_collection = bpy.data.collections.new("Targets")
            bpy.context.scene.collection.children.link(target_collection)

        #  Remove any existing targets first
        for obj in list(target_collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

        # Create new target
        bpy.ops.mesh.primitive_uv_sphere_add(location=(locationX, locationY, 1))
        self.target = bpy.context.active_object

        # Unlink from ALL collections first (to avoid duplicate linking errors)
        for col in self.target.users_collection:
            col.objects.unlink(self.target)

        # Now safely link to Targets collection
        target_collection.objects.link(self.target)


# ===================================================
# MAIN TRAIN LOOP
# ===================================================
if __name__ == "__main__":
    env = pelletGrabberEnv()
    states = env.observation_space.shape[0]
    actions = env.action_space.n

    print(f"Observation space: {states} | Action space: {actions}")

    episodes = 10
    current_episode = 1
    done = True   #  start True so we reset before first episode
    episode_score = 0
    total_score = 0   #  keeps track of all rewards across episodes

    def train():
        global current_episode, done, episode_score, total_score

        if current_episode > episodes:
            print(f"\n🏁 Training finished! Total score across all episodes: {total_score}")
            return None

        if done:
            state, info = env.reset()
            done = False
            episode_score = 0
            print(f"\n--- 🎬 Starting Episode {current_episode} ---")

        action = env.action_space.sample()  # take random action
        n_state, reward, terminated, truncated, info = env.step(action)

        episode_score += reward
        total_score += reward
        done = terminated or truncated

        print(f"Episode {current_episode} | Action: {action} | Reward: {reward} | "
            f"Episode Total: {episode_score} | Overall Total: {total_score}")

        if done:
            print(f" Episode {current_episode} finished with score {episode_score}")
            current_episode += 1

        return 0.1


    #  Start training smoothly
    bpy.app.timers.register(train)


