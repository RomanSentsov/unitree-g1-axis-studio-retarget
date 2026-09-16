import pybvh
from pybvh import bvhplot
from pin_opti import G1_29_ArmIK
import pinocchio as pin
import numpy as np
import time

from scipy.spatial.transform import Rotation as R

# TODO: elbow calibrate.

# ------------------------------ BVH OPEN ------------------------------

FILEPATH = "recs/bvh_recs/bvh_misha_no_xyz_chr01_MAYA.bvh"
# FILEPATH = "recs/bvh_recs/robot_67_cry_joy_wings_photo_pose.bvh"
FILEPATH_EX = "recs/bvh_recs/example1.bvh"

bvh = pybvh.read_bvh_file(FILEPATH)
bvh = bvh.reorient_world_up('+z')
bvh = bvh.rotate_vertical(np.pi / 2)
# bvh.play()

# print(bvh.joint_names)
# exit()


# get indexes of joints
r_hand_idx = bvh.node_index["RightHand"]
l_hand_idx = bvh.node_index["LeftHand"]

r_mid_idx = bvh.node_index["RightHandMiddle1"]
l_mid_idx = bvh.node_index["LeftHandMiddle1"]

r_pinky_idx = bvh.node_index["RightHandPinky1"]
l_pinky_idx = bvh.node_index["LeftHandPinky1"]

r_elbow_idx = bvh.node_index["RightForeArm"]
l_elbow_idx = bvh.node_index["LeftForeArm"]

# Get positions

poses = bvh.node_positions(centered="skeleton")

print(poses.shape)

poses = np.array(poses[::3, :, :])

print(poses.shape)

frame_num = poses.shape[0]

# Relative poses of finger fragments representig hand orient

r_hand_x = poses[:, r_mid_idx] - poses[:, r_hand_idx]
r_hand_y = poses[:, r_pinky_idx] - poses[:, r_mid_idx]

l_hand_x = poses[:, l_mid_idx] - poses[:, l_hand_idx]
l_hand_y = poses[:, l_pinky_idx] - poses[:, l_mid_idx]

quat_list = [R.align_vectors(np.stack([r_hand_x[i], r_hand_y[i]], axis=0), 
                              np.array([[1, 0, 0], [0, 0, -1]]))[0].as_quat() 
             for i in range(len(r_hand_x))]
r_quat = np.array(quat_list)

quat_list = [R.align_vectors(np.stack([l_hand_x[i], l_hand_y[i]], axis=0), 
                              np.array([[1, 0, 0], [0, 0, -1]]))[0].as_quat() 
             for i in range(len(l_hand_x))]
l_quat = np.array(quat_list)
# ------------------------------ Play BVH via retarget  ------------------------------

arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = True)

    # initial positon
R_prev_target = pin.SE3(
        pin.Quaternion(r_quat[0]),
        poses[0, r_hand_idx]/100,
    )

L_prev_target = pin.SE3(
        pin.Quaternion(l_quat[0]),
        poses[0, l_hand_idx]/100,
    )



until = None
now = None

while True:

    for i in range(frame_num - 1):
    #   for i in range(1):
        R_tf_target = pin.SE3(
            pin.Quaternion(r_quat[i]),
            poses[i, r_hand_idx]/150,
        )   

        L_tf_target = pin.SE3(
            pin.Quaternion(l_quat[i]),
            poses[i, l_hand_idx]/150,
        )


        R_tf_elbow_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            poses[i, r_elbow_idx]/150,
        )   

        L_tf_elbow_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            poses[i, l_elbow_idx]/150,
        )

        until = time.time()
        arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, L_tf_elbow_target.homogeneous, R_tf_elbow_target.homogeneous)
        now = time.time()
        elapsed = now - until
        print(now - until)

        time.sleep(np.max([0.033 - elapsed, 0.001]))

    