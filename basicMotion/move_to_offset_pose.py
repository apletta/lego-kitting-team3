import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm


if __name__ == "__main__":
    fa = FrankaArm()
    # fa.reset_pose()
    # fa.reset_joints()
    

    p0 = fa.get_pose()
    # j0 = fa.get_joints()
    print("p0:\n", p0)

    Z_RESET_TO_WORKPLANE = 0.476304483
    p1 = p0.copy()
    # print("j0: \n", j0)    

    T_delta = RigidTransform(
        translation=np.array([0.0, 0.0,0.0]),
        rotation=RigidTransform.z_axis_rotation(np.deg2rad(0)), 
                            from_frame=p0.from_frame, to_frame=p0.from_frame)
        # rotation = np.array([[1., 0., 0.],[0.,1.,0.],[0.,0.,1.]]),
        #                 from_frame=p0.from_frame, to_frame=p1.from_frame)
        
    p1 = p1 * T_delta
    
    fa.goto_pose(p1)


    print("p1:\n", p1)
    
    # j1 = fa.get_joints()


    # print("j1: \n", j1)

    