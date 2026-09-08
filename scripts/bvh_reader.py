import pybvh
from pybvh import bvhplot
from pin_opti import G1_29_ArmIK
import pinocchio as pin
import numpy as np
import time

# ------------------------------ BVH OPEN ------------------------------

FILEPATH = "recs/bvh_recs/bvh_misha_no_xyz_chr01_MAYA.bvh"
FILEPATH_EX = "recs/bvh_recs/example1.bvh"

bvh = pybvh.read_bvh_file(FILEPATH)
bvh = bvh.reorient_world_up('+z')
bvh = bvh.rotate_vertical(np.pi / 2)
# bvh.play()

# print(bvh.joint_names)


# RightHand, LeftHand
r_hand_idx = bvh.node_index["RightHand"]
l_hand_idx = bvh.node_index["LeftHand"]

r_hand_pos = bvh.node_positions(centered="skeleton")
l_hand_pos = bvh.node_positions(centered="skeleton")

print(r_hand_pos.shape)

r_hand_pos = r_hand_pos[::3, :, :]
l_hand_pos = l_hand_pos[::3, :, :]

print(r_hand_pos.shape)

frame_num = r_hand_pos.shape[0]

# ------------------------------ Play BVH via retarget  ------------------------------

arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = True)

    # initial positon
L_prev_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array(l_hand_pos[0, l_hand_idx])/200,
    )

R_prev_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array(r_hand_pos[0, r_hand_idx])/200,
    )

# bvh.play()
# exit()

until = None
now = None

while True:

    for i in range(frame_num - 1):

        L_tf_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array(l_hand_pos[i, l_hand_idx])/200,
        )

        R_tf_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array(r_hand_pos[i, r_hand_idx])/200,
        )   

        until = time.time()
        arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)
        now = time.time()
        elapsed = now - until
        print(now - until)

        time.sleep(np.max([0.033 - elapsed, 0.001]))

    