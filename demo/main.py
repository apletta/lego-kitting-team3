from cv_bridge import CvBridge
from autolab_core import RigidTransform
from perception import CameraIntrinsics
from utils import *
from frankapy import FrankaArm
import numpy as np
from BlockLoc import *
import cv2

class LegoBot:
    def __init__(self):
        # Calibration files
        self.AZURE_KINECT_INTRINSICS = 'calib/azure_kinect.intr'
        self.AZURE_KINECT_EXTRINSICS = 'calib/azure_kinect_overhead_to_world.tf'

        # See calib/azure_kinect.intr
        fx = 973.626953125
        fy = 973.331298828125
        cx = 1020.7919311523438
        cy = 781.0071411132812
        intrinsics = np.array([[fx,0,cx],
                               [0,fy,cy],
                               [0,0,1]])

        r11 = 0.025409
        r12 = 0.999638
        r13 = -0.0088
        r21 = 0.999499
        r22 = -0.025237
        r23 = 0.019097
        r31 = 0.018868 
        r32 = -0.00928 
        r33 = -0.999779
        tx = 0.614567
        ty = -0.014707
        tz = 0.854953
        extrinsics = np.array([[r11,r12,r13,tx],
                               [r21,r22,r23,ty],
                               [r31,r32,r33,tz]]) 
        self.P = np.matmul(intrinsics, extrinsics)

        # Initialize franka arm on startup
        self.fa = FrankaArm()
        self.fa.reset_pose()
        self.fa.reset_joints()
        self.fa.open_gripper()

        # Brick constants, in world frame
        self.Z_PICKUP = 0.003
        self.Z_INTERMEDIATE = 0.2

        # Basket constants, in world frame
        self.X_BASKET = 0.5
        self.Y_BASKET = 0.3
        self.Z_BASKET = 0.2
        self.YAW_BASKET_DEG = 0
        print("LegoBot initialized!")

    def transform_image_to_world(self, px, py):
        # Homogenous image coordinate
        UVH = np.array([px, py,1])

        # Use pseudo inverse to back calculate 3d coordinate
        print(self.P.shape)
        XYZH = np.linalg.pinv(self.P) @ UVH

        # Scale x and y positions by z, for depth ambiguity (hardcoding workspace depth anyway)
        X = XYZH[0]/XYZH[-1]
        Y = XYZH[1]/XYZH[-1]
        print(f"Projected: {XYZH}")
        return X, Y

    def pickup_brick(self, brick_x, brick_y, brick_yaw_rad):

        # Get current pose
        object_center_pose = self.fa.get_pose()

        # Modify current translation to desired translation
        object_center_pose.translation = [brick_x, brick_y, self.Z_PICKUP]

        # Modify current yaw rotation to desired yaw
        # theta = (brick_yaw_deg/180.0)*np.pi
        theta = brick_yaw_rad
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

    def leggo(self):
        # Get current pose
        basket_pose = self.fa.get_pose()

        # Modify current translation to desired translation
        basket_pose.translation = [self.X_BASKET, self.Y_BASKET, self.Z_BASKET]

        # Modify current yaw rotation to desired yaw
        theta = (self.YAW_BASKET_DEG/180.0)*np.pi
        new_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [-np.sin(theta), -np.cos(theta), 0],
                            [0, 0, -1]])
        basket_pose.rotation = new_rotation

        # Move to basket pose
        self.fa.goto_pose(basket_pose, duration=5, force_thresholds=[10, 10, 10, 10, 10, 10])
        
        # Open Gripper
        self.fa.open_gripper()

    def draw_detections(self, frame, blocks_red, blocks_blue):
        RAD_TO_DEG = 180 / np.pi

        for block in blocks_red:
            rect = ((block.x, block.y), (block.length, block.width), block.angle * RAD_TO_DEG)
            box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        for block in blocks_blue:
            rect = ((block.x, block.y), (block.length, block.width), block.angle * RAD_TO_DEG)
            box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

if __name__ == '__main__':    # time.sleep(5.0)

    # move_to_pose(x=0.5, y=0, yaw_deg=0)
    print("Initializing LegoBot...")
    my_eggo = LegoBot()
    print("LegoBot ready!")
    
    print("Loaded camera intrinsics and extrinsics...")
    azure_kinect_intrinsics = CameraIntrinsics.load(my_eggo.AZURE_KINECT_INTRINSICS)
    azure_kinect_to_world_transform = RigidTransform.load(my_eggo.AZURE_KINECT_EXTRINSICS)
    print("Loaded camera intrinsics and extrinsics!")

    # Get images
    print("Getting images...")
    cv_bridge = CvBridge()
    azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
    azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
    print("Got images!")

    # Get block detections
    print("Detecting blocks...")
    frame = azure_kinect_rgb_image[200:850, 500:1300, :3] #cv2.cvtColor(azure_kinect_rgb_image[200:850, 500:1300], cv2.COLOR_BGR2RGB)
    # img[0:420,330:800,:] # BGR

    blocks_red, blocks_blue = get_block_locs(frame)
    my_eggo.draw_detections(frame, blocks_red, blocks_blue)
    print("Blocks detected!")

    print("RED\n-------")
    for block in blocks_red:
        print("{} block: ({},{}); {} x {}; {} rad, {} deg".format(block.color, block.x, block.y, block.length, block.width, block.angle, block.angle*180/np.pi))
    print("BLUE\n-------")
    for block in blocks_blue:
        print("{} block: ({},{}); {} x {}; {} rad, {} deg".format(block.color, block.x,
              block.y, block.length, block.width, block.angle, block.angle * 180 / np.pi))

    # Select target block
    target = blocks_red[0]
    print(f"Target\nx: {target.x}, y: {target.y}, yaw: {target.angle}")

    # Transform from camera frame to world frame
    target_world_frame = get_object_center_point_in_world(int(target.x),
                                                            int(target.y),
                                                            azure_kinect_depth_image, azure_kinect_intrinsics,
                                                            azure_kinect_to_world_transform)  
    # brick_x, brick_y = my_eggo.transform_image_to_world(target.x, target.y)
    print(f"Transformed\nx: {target_world_frame[0]}, y: {target_world_frame[1]}, yaw: {target.angle}")

    cv2.imshow('Webcam', frame)
    cv2.waitKey(0)

    # Pickup the brick and put in basket
    print("Grabbing brick...")
    my_eggo.pickup_brick(target_world_frame[0], target_world_frame[1], brick_yaw_rad=target.angle)

    print("Dropping brick...")
    my_eggo.leggo() # Let go of brick in basket
    