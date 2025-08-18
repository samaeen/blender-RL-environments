import bpy
from math import radians
import time

# --- Clean Scene ---
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# --- Define sizes ---
cartwidth, cartheight = 2.0, 1.0
polewidth, polelen = 0.5, 3.0
axleoffset = cartheight / 4.0

# --- Create Cart ---
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, .25))
cart = bpy.context.active_object
cart.name = "Cart"
cart.scale = (cartwidth/2, cartwidth/2, cartheight/2)

# --- Create Pole ---
pole_z = cartheight + axleoffset
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, pole_z))
pole = bpy.context.active_object
pole.name = "Pole"
pole.scale = (polewidth/2, polewidth/2, polelen/2)

# ✅ Parent Pole to Cart
pole.parent = cart

# ✅ Track simulation start
start_time = time.time()

# ✅ This just MOVES cart & pole by 1 step
def render_step():
    """Moves cart +1 X and rotates pole +1 degree."""
    cart.location.x += 1
    pole.rotation_euler.x += radians(1)

    frame = bpy.context.scene.frame_current
    cart.keyframe_insert(data_path="location", frame=frame)
    pole.keyframe_insert(data_path="rotation_euler", frame=frame)

    bpy.context.scene.frame_set(frame + 1)
    print(f"🎬 Frame {frame}: Cart moved, Pole rotated")


# ✅ This controls TIMING (stops after 3 sec)
def timer_control():
    elapsed = time.time() - start_time

    if elapsed > 3:
        print("✅ Simulation ended after 3 seconds.")
        return None  # stops the timer

    # 🔄 call render once each tick
    render_step()

    # Schedule this function again in 0.1 sec
    return 0.1


# ✅ Start the timer (auto calls every 0.1 sec)
bpy.app.timers.register(timer_control)
