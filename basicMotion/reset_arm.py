from frankapy import FrankaArm

if __name__ == '__main__':
    fa = FrankaArm()
    
    print("Resetting arm to home pose, joint angles, and with gripper open")
    fa.reset_pose()
    fa.reset_joints()
    fa.open_gripper()
    

    p0 = fa.get_pose()
    print(p0)
    j0 = fa.get_joints()
    print(j0)
    fa.open_gripper()