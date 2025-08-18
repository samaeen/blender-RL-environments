from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import bpy
import math
from mathutils import Vector,Euler

class pelletGrabberEnv(Env):
    def __init__(self):
        #------------------Reinforcement Learning Parameters-----------
        # Actions we can take, move right,left, up ,down
        self.action_space = Discrete(4)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        #--------------------Robot Parameters--------------------------
        # PARAMETERS
        self.wheel_speed = 2.0       # Speed of each wheel (m/s)
        self.change_interval = 30 

        #blender robot object configuration
        # Get the robot object
        self.robot = bpy.data.objects.get("Robot")
        if self.robot is None:
            raise Exception("No object named 'Robot' found!")
        
        #  Materials
        self.greenMat = bpy.data.materials.new(name="Green")
        self.greenMat.use_nodes = True
        self.greenMat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0, 1, 0, 1)

        self.redMat = bpy.data.materials.new(name="Red")
        self.redMat.use_nodes = True
        self.redMat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 0, 0, 1)


        #generateTarget
        self.generateTarget()


        

    def step(self,action):
        assert self.state is not None, "Call reset before using step method."
        
        robot_x,robot_y,robot_yaw,raycast_dist1, raycast_dist2,raycast_dist3,raycast_dist4 ,raycast_dist5,reward=self.robotMove(action)
        
        self.current_steps += 1
        if reward != 0 or self.current_steps > self.max_steps:
            terminated = True


        self.state = [
            robot_x, robot_y, robot_yaw,
            raycast_dist1, raycast_dist2, raycast_dist3, raycast_dist4, raycast_dist5
        ]
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
    
    def reset(self):
        self.robot.location = Vector((0, 0, self.robot.dimensions.z))
        bpy.data.objects.remove(self.target, do_unlink=True)
        self.robot.rotation_euler.z = 0.0

        self.generateTarget()

        # initialize state (use dummy values)
        self.state = [self.robot.location.x, self.robot.location.y, self.robot.rotation_euler.z,
                    10.0, 10.0, 10.0, 10.0, 10.0]  # rays start with max distance
        
        return np.array(self.state, dtype=np.float32), {}

    
    def render(self, mode="human"):
        print(f"Robot position: {self.robot.location}, yaw: {self.robot.rotation_euler.z}")

        
    def robotMove(self,action):
        # Use object width (Y dimension) as wheel base
        wheel_base = self.robot.dimensions.y

        if action is not None:
            wheel_base = self.robot.dimensions.y

            if action == 0:       # turn left
                v_l = -self.wheel_speed
                v_r = self.wheel_speed
            elif action == 1:     # turn right
                v_l = self.wheel_speed
                v_r = -self.wheel_speed
            elif action == 2:     # forward
                v_l = self.wheel_speed
                v_r = self.wheel_speed
            elif action == 3:     # backward
                v_l = -self.wheel_speed
                v_r = -self.wheel_speed

            reward, r1, r2, r3, r4, r5 = self.collide()

            # Differential drive equations
            v = (v_r + v_l) / 2.0
            omega = (v_r - v_l) / wheel_base
            
            # Get yaw rotation (Z axis)
            yaw = self.robot.rotation_euler.z
            
            # Update location and rotation
            self.robot.location.x += v * math.cos(yaw) 
            self.robot.location.y += v * math.sin(yaw) 
            self.robot.rotation_euler.z += omega 

        plane = bpy.data.objects.get("Plane")

        #  Collections
        target_collection = bpy.data.collections.get("Targets")
        wall_collection = bpy.data.collections.get("Walls")
        if not target_collection or not wall_collection:
            raise Exception("Collection 'Target' or 'Wall' not found!")

        #  Raycast setup
        num_rays = 5
        arc_angle = math.radians(30)
        max_distance = 10.0
        reward_distance = 0.25

        forward_dir = self.robot.matrix_world.to_quaternion() @ Vector((0, 1, 0))
        start_angle = -arc_angle / 2

        hit_reward = 0

        #  Empty slots for distances (start with None)
        ray1 = ray2 = ray3 = ray4 = ray5 = max_distance
        ray_slots = [max_distance, max_distance, max_distance, max_distance, max_distance]

        for i in range(num_rays):
            angle = start_angle + (i / (num_rays - 1)) * arc_angle
            rotation = Euler((0, 0, angle), 'XYZ')
            ray_dir = rotation.to_matrix() @ forward_dir

            #  Perform raycast
            ray_result = self.robot.ray_cast(
                self.robot.matrix_world.translation,
                self.robot.matrix_world.translation + ray_dir * max_distance
            )

            if len(ray_result) == 6:
                result, location, normal, index, obj, matrix = ray_result
            elif len(ray_result) == 5:
                # Older Blender API: no result flag
                location, normal, index, obj, matrix = ray_result
                result = obj is not None  # fake a hit result
            elif len(ray_result) == 4:
                # Very old Blender/UPBGE (returns only hit, loc, normal, face)
                result, location, normal, index = ray_result
                obj = None
                matrix = None
            else:
                raise RuntimeError(f"Unexpected ray_cast return length: {len(ray_result)}")

            if result and obj:
                hit_distance = (location - self.robot.location).length
                ray_slots[i] = hit_distance  # store distance in proper slot

                #  reward/penalty logic
                if hit_distance <= reward_distance:
                    if obj in target_collection.objects:
                        if not plane.data.materials:
                            plane.data.materials.append(self.greenMat)
                        else:
                            plane.data.materials[0] = self.greenMat
                        hit_reward = 1
                    elif obj in wall_collection.objects:
                        if not plane.data.materials:
                            plane.data.materials.append(self.redMat)
                        else:
                            plane.data.materials[0] = self.redMat
                        hit_reward = -1
            else:
                ray_slots[i] = max_distance  # no hit

        #  Unpack into individual values
        ray1, ray2, ray3, ray4, ray5 = ray_slots
        return hit_reward, ray1, ray2, ray3, ray4, ray5


    
    def generateTarget(self):
        locationX = random.randint(0, 18)
        locationY = random.randint(0, 18)

        target_collection = bpy.data.collections.get("Targets")
        if target_collection is None:
            target_collection = bpy.data.collections.new("Targets")
            bpy.context.scene.collection.children.link(target_collection)

        # Create new target
        bpy.ops.mesh.primitive_uv_sphere_add(location=(locationX, locationY, 1))
        self.target = bpy.context.active_object

        # Unlink from ALL collections first (to avoid duplicate linking errors)
        for col in self.target.users_collection:
            col.objects.unlink(self.target)

        # Now safely link to Targets collection
        target_collection.objects.link(self.target)



if __name__ == "__main__":
    env=pelletGrabberEnv()
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

        #  Start new episode if needed
        if done:
            state, info = env.reset()
            done = False
            score = 0

        #  TAKE ACTION (now all 4 actions possible)
        env.render()
        action = env.action_space.sample()  # (0: left, 1: right, 2: forward, 3: back)

        #  MOVE ROBOT
        n_state, reward, terminated, truncated, info = env.step(action)

        #  TRACK REWARD
        score += reward
        done = terminated or truncated

        print(f"Episode {current_episode} | Action: {action} | Reward: {reward} | Total: {score}")

        #  RESET EPISODE IF FINISHED
        if done:
            print(f" Episode {current_episode} finished with score {score}")
            current_episode += 1
            done = True   # next call will reset robot & spawn pellet

        return 0.1  # call again after 0.1s

    # Start training smoothly
    bpy.app.timers.register(train)