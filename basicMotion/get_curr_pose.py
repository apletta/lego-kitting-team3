import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm


if __name__ == "__main__":
    fa = FrankaArm()   
    curr_pose = fa.get_pose()
    print(curr_pose)