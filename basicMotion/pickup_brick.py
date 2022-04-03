import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm

# Gripper constants
GRIPPER_WIDTH_MAX = 0.08
GRIPPER_WIDTH_MIN = 0
GRIPPER_MAX_FORCE = 60

# Motion constants
Z_RESET_TO_WORKPLANE = 0.476304483

# Basket constants = offset from home pose to basket pose
RESET_TO_BASKET = [0.15, 0.26, 0.3] # [tx, ty, tz]


# Initialize franka arm
fa = FrankaArm()
fa.reset_pose()
fa.reset_joints()
fa.open_gripper()


def move_to_brick(tx, ty, rz_deg):
    # Start from home position
    fa.reset_pose()
    fa.reset_joints()

    p0 = fa.get_pose()
    p1 = p0.copy()

    T_delta = RigidTransform(
            translation=np.array([tx, ty, Z_RESET_TO_WORKPLANE]),
            rotation=RigidTransform.z_axis_rotation(np.deg2rad(rz_deg)), 
                                from_frame=p0.from_frame, to_frame=p0.from_frame)

    p1 = p1 * T_delta  
    fa.goto_pose(p1)

def grab_brick():
    fa.goto_gripper(GRIPPER_WIDTH_MIN, grasp=True, force=GRIPPER_MAX_FORCE*0.5)

def move_to_basket():
    # Start from home position
    fa.reset_pose()
    fa.reset_joints()

    p0 = fa.get_pose()
    p1 = p0.copy()

    T_delta = RigidTransform(
            translation=np.array(RESET_TO_BASKET),
            rotation=RigidTransform.z_axis_rotation(np.deg2rad(0)), 
                                from_frame=p0.from_frame, to_frame=p0.from_frame)

    p1 = p1 * T_delta  
    fa.goto_pose(p1)

if __name__ == "__main__":

    # Move to home then to position over brick
    print("Moving to brick...")
    move_to_brick(tx=0.1, ty=0, rz_deg=0)

    # Close grippers on the brick
    print("Grabbing brick...")
    grab_brick()

    # Move to home then to over basket while holding brick
    print("Carrying brick to basket...")
    move_to_basket()
    
    # Open gripper to drop brick in basket
    print("Dropping brick...")
    fa.open_gripper()
    