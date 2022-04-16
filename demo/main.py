from cv_bridge import CvBridge
from autolab_core import RigidTransform
from perception import CameraIntrinsics
from utils import *
from frankapy import FrankaArm
import numpy as np
import time

class LegoBot:
    def __init__(self):
        self.AZURE_KINECT_INTRINSICS = 'calib/azure_kinect.intr'
        self.AZURE_KINECT_EXTRINSICS = 'calib/azure_kinect_overhead_to_world.tf'

        # Initialize franka arm
        self.fa = FrankaArm()
        self.fa.reset_pose()
        self.fa.reset_joints()
        self.fa.open_gripper()

        self.Z_RESET_TO_WORKPLANE = 0.476304483
        self.Z_PICKUP = 0.003
        self.Z_INTERMEDIATE = 0.2


    def pickup_brick(self, brick_x, brick_y, brick_yaw_deg):

        # Get current pose
        object_center_pose = self.fa.get_pose()

        # Modify current translation to desired translation
        object_center_pose.translation = [brick_x, brick_y, self.Z_PICKUP]

        # Modify current yaw rotation to desired yaw
        theta = (brick_yaw_deg/180.0)*np.pi
        new_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [-np.sin(theta), -np.cos(theta), 0],
                            [0, 0, -1]])
        object_center_pose.rotation = new_rotation

        # Create intermediate pose
        intermediate_robot_pose = object_center_pose.copy()
        intermediate_robot_pose.translation = [brick_x, brick_y, self.Z_INTERMEDIATE]

        # Move to intermediate pose
        self.fa.goto_pose(intermediate_robot_pose)

        # Move to object pose
        self.fa.goto_pose(object_center_pose, duration=5, force_thresholds=[10, 10, 10, 10, 10, 10])
        
        # Close Gripper
        self.fa.goto_gripper(width=0.045, grasp=True, force=10.0)

    # def drop_in_kit

if __name__ == '__main__':    # time.sleep(5.0)

    # move_to_pose(x=0.5, y=0, yaw_deg=0)


    # time.sleep(5.0)

    # move_to_pose(x=0.6, y=0, yaw_deg=0)get_azure_kinect_rgb_image(cv_bridge)
    # azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)


    # object_center_point_in_world = get_object_center_point_in_world(x_pos,
    #                                                                 y_pos,
    #                                                                 azure_kinect_depth_image, azure_kinect_intrinsics,
    #                                                                 azure_kinect_to_world_transform)  

    move_to_pose(x=0.4, y=0, yaw_deg=0)

    # time.sleep(5.0)

    move_to_pose(x=0.5, y=0, yaw_deg=0)


    # time.sleep(5.0)

    # move_to_pose(x=0.6, y=0, yaw_deg=0)