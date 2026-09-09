import pybvh
from pybvh import bvhplot
from pin_opti import G1_29_ArmIK
import pinocchio as pin
import numpy as np
import time

from scipy.spatial.transform import Rotation as R
from bvh import Bvh
# ------------------------------ BVH OPEN ------------------------------

FILEPATH = "recs/bvh_recs/bvh_misha_no_xyz_chr01_MAYA.bvh"
FILEPATH_EX = "recs/bvh_recs/example1.bvh"


bvh = Bvh()
bvh.parse_file(FILEPATH)

positions, rotations = bvh.all_frame_poses()

# for i, _p, _r, _j in zip(range(65), positions[2], rotations[2], bvh.joint_names()):
#         print(f"{i}: {_j}: p={_p}, r={_r}")

# exit()

# RightHand 20, LeftHand 44
# RightHandMiddle1 29, LeftHandMiddle1 53
# RightHandIndex1 25, LeftHandIndex1 49
positions = positions[::3, :, :]
rotations = rotations[::3, :, :]



r_hand_pos = positions[:, 20, :]
r_hand_rot = rotations[:, 20, :]
l_hand_pos = positions[:, 44, :]
l_hand_rot = rotations[:, 44, :]

base_pos = positions[:, 0, :]
base_rot = rotations[:, 0, :]

r_fing_pos = positions[:, 29, :]
l_fing_pos = positions[:, 53, :]

r_fing1_pos = positions[:, 25, :]
l_fing1_pos = positions[:, 49, :]

# print(r_hand_pos.shape)

frame_num = r_hand_pos.shape[0]

r_hand_pos_rel = []
r_hand_rot_rel = []
l_hand_pos_rel = []
l_hand_rot_rel = []

r_fing_pos_rel = []
l_fing_pos_rel = []

r_fing1_pos_rel = []
l_fing1_pos_rel = []

r_orient_vec = []
l_orient_vec = []

r_orient1_vec = []
l_orient1_vec = []

for i in range(frame_num):

    r_hand_pos_rel.append(np.array(r_hand_pos[i]) - np.array(base_pos[i]))
    # r_hand_pos_rel[i] = (R.from_euler('z', np.pi/2)).apply(r_hand_pos_rel[i])
    r_hand_pos_rel[i] = (R.from_euler('y', np.pi/2)).apply(r_hand_pos_rel[i])
    r_hand_pos_rel[i] = (R.from_euler('x', np.pi/2)).apply(r_hand_pos_rel[i])
    r_hand_rot_rel.append(r_hand_rot[i])

    l_hand_pos_rel.append(np.array(l_hand_pos[i]) - np.array(base_pos[i]))
    # l_hand_pos_rel[i] = (R.from_euler('z', np.pi/2)).apply(l_hand_pos_rel[i])
    l_hand_pos_rel[i] = (R.from_euler('y', np.pi/2)).apply(l_hand_pos_rel[i])
    l_hand_pos_rel[i] = (R.from_euler('x', np.pi/2)).apply(l_hand_pos_rel[i])
    l_hand_rot_rel.append(l_hand_rot[i])

    # ---- for fingers the same

    r_fing_pos_rel.append(np.array(r_fing_pos[i]) - np.array(base_pos[i]))
    r_fing_pos_rel[i] = (R.from_euler('y', np.pi/2)).apply(r_fing_pos_rel[i])
    r_fing_pos_rel[i] = (R.from_euler('x', np.pi/2)).apply(r_fing_pos_rel[i])

    l_fing_pos_rel.append(np.array(l_fing_pos[i]) - np.array(base_pos[i]))
    l_fing_pos_rel[i] = (R.from_euler('y', np.pi/2)).apply(l_fing_pos_rel[i])
    l_fing_pos_rel[i] = (R.from_euler('x', np.pi/2)).apply(l_fing_pos_rel[i])

    # ---- for index fing to define orientation of palm

    r_fing1_pos_rel.append(np.array(r_fing1_pos[i]) - np.array(base_pos[i]))
    r_fing1_pos_rel[i] = (R.from_euler('y', np.pi/2)).apply(r_fing1_pos_rel[i])
    r_fing1_pos_rel[i] = (R.from_euler('x', np.pi/2)).apply(r_fing1_pos_rel[i])

    l_fing1_pos_rel.append(np.array(l_fing1_pos[i]) - np.array(base_pos[i]))
    l_fing1_pos_rel[i] = (R.from_euler('y', np.pi/2)).apply(l_fing1_pos_rel[i])
    l_fing1_pos_rel[i] = (R.from_euler('x', np.pi/2)).apply(l_fing1_pos_rel[i])
    # ----------- orient vecs

    r_orient_vec.append(r_fing_pos_rel[i] - r_hand_pos_rel[i])
    l_orient_vec.append(l_fing_pos_rel[i] - l_hand_pos_rel[i])

    r_orient1_vec.append(r_fing1_pos_rel[i] - r_fing_pos_rel[i])
    l_orient1_vec.append(l_fing1_pos_rel[i] - l_fing_pos_rel[i])

    
# ------------------------------ Play BVH via retarget  ------------------------------

arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = True)

    # initial positon
L_prev_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array(l_hand_pos_rel[0])/200,
    )

R_prev_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array(r_hand_pos_rel[0])/200,
    )

# bvh.play()
# exit()

until = None
now = None

while True:

    for i in range(frame_num - 1):
    # for i in range(1):
        vector_norm = r_orient_vec[i] / np.linalg.norm(r_orient_vec[i])
        rotation, _ = R.align_vectors([[-1, 0, 0]], [vector_norm])
        quat_r = rotation.as_quat()

        vector_norm = l_orient_vec[i] / np.linalg.norm(l_orient_vec[i])
        rotation, _ = R.align_vectors([[-1, 0, 0]], [vector_norm])
        quat_l = rotation.as_quat()


        vector_norm = r_orient1_vec[i] / np.linalg.norm(r_orient1_vec[i])
        rotation, _ = R.align_vectors([[0, 0, 1]], [vector_norm])
        quat_r1 = rotation.as_quat()

        vector_norm = l_orient1_vec[i] / np.linalg.norm(l_orient1_vec[i])
        rotation, _ = R.align_vectors([[0, 0, 1]], [vector_norm])
        quat_l1 = rotation.as_quat()

        quat_r = quat_r * quat_r1
        quat_l = quat_l * quat_l1


        L_tf_target = pin.SE3(
            pin.Quaternion(quat_l),
            np.array(l_hand_pos_rel[i])/100,
        )

        R_tf_target = pin.SE3(
            pin.Quaternion(quat_r),
            np.array(r_hand_pos_rel[i])/100,
        )   

        until = time.time()
        arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)
        now = time.time()
        elapsed = now - until
        print(now - until)

        time.sleep(np.max([0.033 - elapsed, 0.001]))

    