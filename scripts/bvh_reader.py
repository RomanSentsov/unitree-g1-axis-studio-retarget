import pybvh
from pybvh import bvhplot
from pin_opti import G1_29_ArmIK
import pinocchio as pin
import numpy as np
import time

from scipy.spatial.transform import Rotation as R

# ------------------------------ BVH OPEN ------------------------------

FRAMERATE = 50
FILEPATH = "recs/bvh_recs/bvh_misha_no_xyz_chr01_MAYA.bvh"
# FILEPATH = "recs/bvh_recs/Aleksandr_recs/rec1_chr01_MAYA.bvh"
FILEPATH_EX = "recs/bvh_recs/example1.bvh"

# reorient to match pinocchio 
bvh = pybvh.read_bvh_file(FILEPATH)
bvh = bvh.reorient_world_up('+z')
bvh = bvh.rotate_vertical(np.pi / 2)
# bvh.play()

# extract node poses
poses = bvh.node_positions(centered="skeleton")
poses = np.array(poses[::3, :, :])
idxes_dict = {name: bvh.node_index[name] for name in bvh.joint_names}
print(poses.shape)
frame_num = poses.shape[0]

# ------------------------------ Play BVH via retarget  ------------------------------

arm_ik = G1_29_ArmIK(Unit_Test = False, Visualization = True)

until = None
now = None

while True:
    for i in range(frame_num):

        # current bvh frame
        bvh_dict = {name: poses[i, idxes_dict[name], :] for name in bvh.joint_names} 

        until = time.time()

        q, tau = arm_ik.solve_ik_bvh_frame(bvh_dict)

        now = time.time()
        elapsed = now - until
        # print(now - until)
        print(q)
        time.sleep(np.max([1/FRAMERATE - elapsed, 0.00001]))

    