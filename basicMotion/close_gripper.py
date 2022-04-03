from frankapy import FrankaArm

# Gripper constants
GRIPPER_WIDTH_MAX = 0.08
GRIPPER_WIDTH_MIN = 0
GRIPPER_MAX_FORCE = 60

if __name__ == '__main__':
    fa = FrankaArm()
    # fa.close_gripper()

    fa.goto_gripper(GRIPPER_WIDTH_MIN, grasp=True, force=GRIPPER_MAX_FORCE)